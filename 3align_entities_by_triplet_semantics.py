# -*- coding: utf-8 -*-
import logging
import json  # 【修改位置 1：新增导入，用于加载别名词典】

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("entity_alignment.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import os
from Levenshtein import distance as edit_distance
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ===== 修改位置 1：加载别名词典 =====
alias_dict_path = r""  # 请确保该路径正确
if os.path.exists(alias_dict_path):
    with open(alias_dict_path, "r", encoding="utf-8") as f:
        alias_dict = json.load(f)
    logger.info("Alias dictionary loaded successfully.")
else:
    alias_dict = {}
    logger.warning("Alias dictionary file not found. Proceeding without it.")

# ===== 修改位置 2：定义归一化和缩写扩展函数 =====
def normalize_entity(entity):
    return entity.lower().strip() if isinstance(entity, str) else ""

def expand_abbreviation(entity):
    norm = normalize_entity(entity)
    # 处理别名词典中键值可能带有引号的情况
    norm_clean = norm.strip('"')
    # 根据别名词典映射，将输入扩展为标准化名称；若无映射则返回归一化后的名称
    return alias_dict.get(f"\"{norm_clean}\"", norm)

# Step 1: 初始化嵌入模型
model = SentenceTransformer("all-MiniLM-L6-v2")  # 可替换为 bge-base-en 等

# Step 2: 读取描述
# 修改后的函数：生成两个拼接字段
def format_semantic_triplet_with_description(row):
    # 标准化字段：若存在标准化字段，则使用之，否则使用原字段
    subj_std = row["subject_standard"] if pd.notna(row["subject_standard"]) else row["subject"]
    obj_std = row["object_standard"] if pd.notna(row["object_standard"]) else row["object"]
    relation = row["relation"]

    # 构造 subject 相关文本：仅使用 subject_description
    subj_desc = row["subject_description"] if pd.notna(row["subject_description"]) else ""
    subject_text = f"Subject: {subj_std}"
    if subj_desc.strip():
        subject_text += f"\nDescription: {subj_desc.strip()}"
    subject_text += f"\nFact: {subj_std} {relation}"

    # 构造 object 相关文本：仅使用 object_description（object 可能为空）
    obj_desc = row["object_description"] if pd.notna(row["object_description"]) else ""
    if obj_std and str(obj_std).strip():
        object_text = f"Object: {obj_std}"
        if obj_desc.strip():
            object_text += f"\nDescription: {obj_desc.strip()}"
        object_text += f"\nFact: {relation} {obj_std}"
    else:
        object_text = ""  # 如果 object 为空则返回空字符串

    # 返回两个字段，后续将添加到 DataFrame 中
    return pd.Series({"semantic_text_subject": subject_text, "semantic_text_object": object_text})

# Step 3: 加载三元组图谱，生成语义向量
def load_semantic_triplets_with_description(csv_path):
    # 修改位置：移除 errors 参数，因为 pd.read_csv() 不支持此参数
    df = pd.read_csv(csv_path, encoding="utf-8-sig", encoding_errors="replace")


    
    # 生成两列语义文本：subject 与 object 两部分
    semantic_df = df.apply(format_semantic_triplet_with_description, axis=1)
    df = pd.concat([df, semantic_df], axis=1)

    # 计算嵌入
    print("Encoding subject semantic texts...")
    subj_embeddings = model.encode(df["semantic_text_subject"].tolist(), convert_to_tensor=True)
    
    print("Encoding object semantic texts...")
    # 对于空字符串（object 为空），模型也能处理；也可选择用占位符替代
    obj_embeddings = model.encode(df["semantic_text_object"].tolist(), convert_to_tensor=True)
    
    return df, subj_embeddings, obj_embeddings

# Step Q: 计算匹配覆盖率  
def compute_matching_coverage(result_df):
    """
    计算匹配覆盖率，包括各匹配类型（exact、edit、semantic、none）的比例
    """
    total = len(result_df)
    coverage = result_df['match_type'].value_counts().to_dict()
    # 将各匹配类型的数量转换为比例
    coverage_rate = {k: v / total for k, v in coverage.items()}
    logger.info(f"Matching Coverage Rates: {coverage_rate}")
    return coverage_rate

# Step 4: 主函数，对齐用户输入实体列表
def align_entities(items, kg_df, subj_embeddings, obj_embeddings, top_k=1, max_edit_dist=2):
    aligned_results = []

    # 构建映射表，使用标准化的 subject 与 object
    alias_set = set(kg_df["subject_standard"]).union(set(kg_df["object_standard"]))

    for item in tqdm(items, disable=True):
        matched = {
            "input": item,
            "match_type": "none",
            "matched_kg_entity": None,
            "matched_triple": None,
            "similarity_score": 0.0,
            "match_source": "none",
            "aligned": False
        }

        # ===== 修改位置 3：使用归一化和别名扩展 =====
        normalized_item = expand_abbreviation(item)
        normalized_item = normalize_entity(normalized_item)

        normalized_alias_set = {normalize_entity(ent) for ent in alias_set}

        # 如果 exact 匹配成立，记录 exact_candidate，但不直接返回
        if normalized_item in normalized_alias_set:
            matched["exact_candidate"] = normalized_item
        else:
            matched["exact_candidate"] = None

        # 2. 编辑距离匹配（使用归一化后的输入）
        closest_alias = None
        min_dist = float("inf")
        for ent in alias_set:
            norm_ent = normalize_entity(ent)
            d = edit_distance(normalized_item, norm_ent)
            if d < min_dist and d <= max_edit_dist:
                closest_alias = ent
                min_dist = d
        if closest_alias:
            matched["edit_candidate"] = closest_alias
        else:
            matched["edit_candidate"] = None

        # 3. 嵌入语义相似度：分别计算 subject 与 object 部分的相似度
        item_vec = model.encode(item, convert_to_tensor=True)
        
        subj_scores = util.cos_sim(item_vec, subj_embeddings)[0].cpu().numpy()
        obj_scores = util.cos_sim(item_vec, obj_embeddings)[0].cpu().numpy()
        
        subj_top_index = int(np.argmax(subj_scores))
        obj_top_index = int(np.argmax(obj_scores))
        
        subj_best_score = float(subj_scores[subj_top_index])
        obj_best_score = float(obj_scores[obj_top_index])
        
        # 根据得分选择匹配源（subject 或 object）
        if subj_best_score >= obj_best_score and subj_best_score > 0.7:
            best_row = kg_df.iloc[subj_top_index]
            matched["match_type"] = "semantic"
            matched["matched_kg_entity"] = best_row["subject_standard"]
            matched["matched_triple"] = best_row["semantic_text_subject"]
            matched["similarity_score"] = subj_best_score
            matched["match_source"] = "subject"
            matched["aligned"] = True
        elif obj_best_score > subj_best_score and obj_best_score > 0.7:
            best_row = kg_df.iloc[obj_top_index]
            matched["match_type"] = "semantic"
            matched["matched_kg_entity"] = best_row["object_standard"]
            matched["matched_triple"] = best_row["semantic_text_object"]
            matched["similarity_score"] = obj_best_score
            matched["match_source"] = "object"
            matched["aligned"] = True
        else:
            matched["match_type"] = "none"
            matched["matched_kg_entity"] = ""
            matched["matched_triple"] = ""
            matched["similarity_score"] = 0
            matched["match_source"] = "none"
            matched["aligned"] = False

        aligned_results.append(matched)

    return pd.DataFrame(aligned_results)

# Step 5: 自动输出对齐比例、未对齐 Top10 实体
from collections import Counter

def summarize_alignment_stats(df, summary_file="alignment_summary.txt"):
    lines = []
    lines.append("Alignment Summary")
    lines.append("-" * 30)
    total = len(df)
    aligned = df["aligned"].sum()
    failed = total - aligned
    lines.append(f"Total entities processed: {total}")
    lines.append(f" Aligned: {aligned}")
    lines.append(f" Not aligned: {failed}")

    lines.append("\n Match type distribution:")
    match_dist = Counter(df["match_type"])
    lines.append(str(match_dist))

    if "similarity_score" in df.columns and aligned > 0:
        avg_score = df[df["aligned"]]["similarity_score"].mean()
        lines.append(f"\n Avg similarity (aligned): {avg_score:.3f}")
    else:
        lines.append("\n No aligned entities to calculate average similarity.")

    unmatched = df[df["aligned"] == False]
    if not unmatched.empty:
        lines.append("\n Top 10 unmatched entities:")
        top10_unmatched = unmatched["input"].value_counts().head(10)
        lines.append(top10_unmatched.to_string())

    summary_str = "\n".join(lines)
    
    logger.info(summary_str)
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_str)
    
    print(summary_str)

# Step 6: 示例调用流程
import datetime  # 新增导入

if __name__ == "__main__":
    triple_file = r" "
    print(os.path.exists(triple_file))

    kg_df, subj_embeds, obj_embeds = load_semantic_triplets_with_description(triple_file)
    
    all_entities = set(kg_df["subject_standard"].dropna()).union(set(kg_df["object_standard"].dropna()))

    result_df = align_entities(list(all_entities), kg_df, subj_embeds, obj_embeds)
    
    # 生成当前日期时间字符串，例如 "2025-04-06_19-15-00"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"aligned_entity_results_{now_str}.csv"
    
    result_df.to_csv(output_filename, index=False)
    logger.info(f"对齐完成，结果保存为 {output_filename}")

    summarize_alignment_stats(result_df)
    compute_matching_coverage(result_df)

    print(f"对齐完成，结果保存为 {output_filename}")
