import os
import pandas as pd
import json
import matplotlib.pyplot as plt

# 文件路径（请确保路径、文件名均正确）
original_csv_path = r""
result_csv_path = r""

# 检查文件是否存在
if not os.path.exists(original_csv_path):
    print(f"错误：原始文件未找到：{original_csv_path}")
    exit(1)
if not os.path.exists(result_csv_path):
    print(f"错误：结果文件未找到：{result_csv_path}")
    exit(1)

# 尝试读取 CSV 文件，遇到编码错误时改用 gbk 编码
try:
    original_df = pd.read_csv(original_csv_path, encoding="utf-8")
except UnicodeDecodeError:
    original_df = pd.read_csv(original_csv_path, encoding="gbk")

try:
    result_df = pd.read_csv(result_csv_path, encoding="utf-8")
except UnicodeDecodeError:
    result_df = pd.read_csv(result_csv_path, encoding="gbk")

# === STEP1: 反写对齐结果到原始图谱文件 ===
aligned = result_df[result_df["aligned"] == True]
alias_map = dict(zip(aligned["input"], aligned["matched_kg_entity"]))

original_df["subject_aligned"] = original_df["subject"].apply(lambda x: alias_map.get(x, x))
original_df["object_aligned"] = original_df["object"].apply(lambda x: alias_map.get(x, x))
original_df.to_csv("updated_triples_with_alignment.csv", index=False)
print("原图谱已添加标准化实体列并保存为 updated_triples_with_alignment.csv")

# === STEP2: 保存 alias 字典 ===
alias_dict = dict(zip(aligned["input"], aligned["matched_kg_entity"]))
with open("aligned_alias_dictionary.json", "w", encoding="utf-8") as f:
    json.dump(alias_dict, f, ensure_ascii=False, indent=2)
print("已保存 alias 映射字典到 aligned_alias_dictionary.json")

# === STEP3: 分析未对齐实体 ===
unmatched = result_df[result_df["aligned"] == False]
print("\nTop 未匹配实体：")
print(unmatched["input"].value_counts().head(20))

# === STEP4: 融合图谱去重 ===
deduplicated_df = original_df.drop_duplicates(subset=["subject_aligned", "relation", "object_aligned"])
# 可继续添加来源图谱字段处理

# === STEP5: 构建融合报告 CSV ===
fusion_report = result_df[["input", "matched_kg_entity", "match_type", "similarity_score", "matched_triple", "match_source", "aligned"]]
fusion_report.to_csv("entity_fusion_report.csv", index=False)
print("已生成融合报告文件 entity_fusion_report.csv")

# === STEP6: 生成对齐摘要报告 ===
total = len(result_df)
aligned_count = result_df["aligned"].sum()
unaligned_count = total - aligned_count
match_counts = result_df["match_type"].value_counts()

# 饼图：匹配方式占比
plt.figure(figsize=(6, 6))
match_counts.plot.pie(autopct='%1.1f%%', startangle=140)
plt.title("Match Type Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("match_type_distribution.png")
plt.close()

# 相似度分布图（仅限已匹配）
plt.figure(figsize=(6, 4))
result_df[result_df["aligned"]]["similarity_score"].hist(bins=20)
plt.title("Similarity Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("similarity_score_histogram.png")
plt.close()

# Top 未匹配实体
top_unmatched = result_df[result_df["aligned"] == False]["input"].value_counts().head(10)

# 输出 Markdown 报告
with open("fusion_summary_report.md", "w", encoding="utf-8") as f:
    f.write("# Entity Alignment Fusion Report\n\n")
    f.write(f"**Total entities processed:** {total}\n\n")
    f.write(f"- ✅ Aligned: `{aligned_count}`\n")
    f.write(f"- ❌ Not aligned: `{unaligned_count}`\n\n")
    f.write("## Match Type Distribution\n\n")
    f.write("![Match Type Distribution](match_type_distribution.png)\n\n")
    f.write("## Similarity Score Histogram (Aligned only)\n\n")
    f.write("![Similarity Score Histogram](similarity_score_histogram.png)\n\n")
    f.write("## Top 10 Unmatched Entities\n\n")
    f.write("```\n")
    f.write(top_unmatched.to_string())
    f.write("\n```\n")

print("📄 已生成融合摘要报告：fusion_summary_report.md 和两张图表")
