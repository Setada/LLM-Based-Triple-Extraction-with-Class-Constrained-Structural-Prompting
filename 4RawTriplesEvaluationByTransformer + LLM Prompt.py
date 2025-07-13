import os
import sys
import time
import json
import re
import spacy
import pandas as pd
from transformers import pipeline
from openai import OpenAI
import difflib


# 设置标准输出为 UTF-8，避免 Unicode 编码问题
sys.stdout.reconfigure(encoding='utf-8')

# 加载 spaCy 英文模型（确保已安装 en_core_web_sm）
nlp = spacy.load("en_core_web_sm")

# === 配置说明 ===
# 1. 三元组数据（JSON 文件）必须包含 "sentence_source" 字段，
#    存放用于定位事故报告中完整句子的子串。
# 2. Context 文件为完整事故报告文本（TXT 格式），
#    利用 spaCy 将其分割为句子列表，通过匹配 "sentence_source" 子串，
#    定位最终用于评估的完整句子。
# 3. 利用 Transformer (roberta-large-mnli) 批量计算蕴含得分，
#    同时调用 LLM 分别对事实准确性、结构连贯性、内容真实三个维度进行评分。
# 4. 输出 CSV 文件包含字段：
#    subject, relation, object, sentence_source, sentence,
#    transformer_score, llm_factual_accuracy, llm_structural_coherence, llm_groundedness.
# 5. 若目标 CSV 文件已存在，则跳过该文件。

# === 配置部分 ===
classifier = pipeline("text-classification", model="roberta-large-mnli")
client = OpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    base_url="",
)
MODEL = " "

def fuzzy_match(target, candidates, threshold=0.7):
    best = max(candidates, key=lambda c: difflib.SequenceMatcher(None, target, c).ratio(), default=None)
    if best and difflib.SequenceMatcher(None, target, best).ratio() >= threshold:
        return best
    return target

# === Transformer 评分函数（基于长文本分句并聚合） ===
def score_long_text_for_transformer(text, triple, aggregation='average'):
    """
    针对长文本，通过 spaCy 分句，将文本分成多个较短句子，
    对每个句子与由 triple 构造的假设句计算蕴含得分，
    最后根据 aggregation 策略聚合各句子的得分，返回整体得分。
    
    参数：
      - text: 长文本（字符串）。
      - triple: 三元组 (subject, relation, object)。
      - aggregation: 聚合策略，默认 'average'（可选 'max'）。
      
    返回：
      - 一个浮点数作为整体蕴含得分（0~1）。
    """
    s, r, o = triple
    hypothesis = f"The {s} {r} the {o}."
    
    # 分句
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    scores = []
    for sent in sentences:
        try:
            # 启用截断并设定最大 token 数量，避免超长输入出错
            result = classifier(sent, text_pair=hypothesis, truncation=True, max_length=512)
            score = 0.0
            for res in result:
                if res['label'] == "ENTAILMENT":
                    score = res['score']
                    break
            scores.append(score)
        except Exception as e:
            print(f"处理句子时出错: {sent[:50]}... 错误: {e}")
            scores.append(0.0)
    if not scores:
        return 0.0
    if aggregation == 'max':
        return max(scores)
    else:
        return sum(scores) / len(scores)

# === LLM 评分函数（更新提示词，分别评分三个维度） ===
def score_with_llm(sentence, triple, retries=3, backoff=1):
    """
    使用 LLM 对给定的 sentence 与 triple 进行事实验证评分，
    分别对以下三个维度打分（分值范围 0 至 3）：
      - 事实准确性 (Factual Accuracy)
      - 结构连贯性 (Structural Coherence)
      - 内容真实 (Groundedness)
      
    同时要求给出简短解释。
    输出必须严格符合如下 JSON 格式：
    {
      "factual_accuracy": <0/1/2/3>,
      "structural_coherence": <0/1/2/3>,
      "groundedness": <0/1/2/3>,
      "explanation": "<concise explanation>"
    }
    遇到异常时重试，所有尝试失败则返回 -1。
    """
    s, r, o = triple
    prompt = f"""
You are a fact verification evaluator.

Please evaluate how well the following sentence supports the given triple by separately scoring the following dimensions:

1. Factual Accuracy: Does the sentence contain information that factually supports the triple? (0 = not supported at all, 1 = partially supported with major issues, 2 = mostly supported with minor issues, 3 = fully supported and faithful)
2. Structural Coherence: Is the sentence structurally coherent in presenting the relation described by the triple? (Same scale: 0 to 3)
3. Groundedness: Is the information in the sentence grounded in reality without hallucination? (Again, 0 to 3)

Provide a brief explanation for each dimension if possible.

Output the result in JSON format exactly as follows:
{{
  "factual_accuracy": <0/1/2/3>,
  "structural_coherence": <0/1/2/3>,
  "groundedness": <0/1/2/3>,
  "explanation": "<concise explanation>"
}}

Do not output anything else.
Sentence: "{sentence}"
Triple: ({s}, {r}, {o})
"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            answer = response.choices[0].message.content.strip()
            try:
                result_dict = json.loads(answer)
                if all(key in result_dict for key in ["factual_accuracy", "structural_coherence", "groundedness"]):
                    return result_dict
            except Exception as json_err:
                print(f"尝试 {attempt+1}: JSON解析失败: {json_err}. 返回内容: {answer}")
        except Exception as e:
            print(f"尝试 {attempt+1}: LLM 调用异常: {e}")
        time.sleep(backoff * (2 ** attempt))
    return -1

# === 主进程 ===
if __name__ == "__main__":
    # triple_dir: 存放三元组 JSON 文件（格式要求为 JSON）
    # context_dir: 存放完整事故报告文本文件（格式要求为 TXT）
    triple_dir = r" "
    context_dir = r" "
    output_dir = r" "
    os.makedirs(output_dir, exist_ok=True)

    error_logs = []  # 用于记录所有错误和异常

    for filename in os.listdir(triple_dir):
        if not filename.endswith(".json"):
            continue

        triple_path = os.path.join(triple_dir, filename)
        # context 文件格式为 TXT
        context_path = os.path.join(context_dir, filename.replace(".json", ".txt"))
        output_path = os.path.join(output_dir, f"scored_{filename.replace('.json', '.csv')}")
        
        # 跳过已存在的输出文件
        if os.path.exists(output_path):
            print(f"输出文件已存在，跳过: {output_path}")
            continue

        if not os.path.exists(context_path):
            print(f"未找到匹配的 context 文件 (应为 TXT 格式): {context_path}")
            continue

        print(f"\n正在处理文件: {filename}")
        try:
            with open(triple_path, "r", encoding="utf-8") as f:
                triple_data = json.load(f)
        except Exception as e:
            print(f"读取 {triple_path} 出现异常: {e}")
            error_logs.append({"file": filename, "error": f"读取异常: {e}"})
            continue

        triples = pd.DataFrame(triple_data)

        if "sentence_source" not in triples.columns:
            msg = f"三元组文件缺少 sentence_source 字段，跳过文件: {filename}"
            print(msg)
            error_logs.append({"file": filename, "error": "三元组缺少 sentence_source 字段"})
            continue

        try:
            with open(context_path, "r", encoding="utf-8") as f:
                context_text = f.read()
        except Exception as e:
            print(f"读取 {context_path} 出现异常: {e}")
            error_logs.append({"file": filename, "error": f"context读取异常: {e}"})
            continue

        # 利用 spaCy 对完整文本进行句子切分
        doc = nlp(context_text)
        context_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # 对每个三元组定位完整句子（利用 "sentence_source" 子串匹配）
        final_sentences = []
        transformer_scores = []
        llm_fa_scores = []
        llm_sc_scores = []
        llm_gr_scores = []

        for i, row in triples.iterrows():
            triple_tuple = (row["subject"], row["relation"], row["object"])
            sentence_source = row["sentence_source"]
            full_sentence = None
            for sent in context_sentences:
                if sentence_source in sent:
                    full_sentence = sent
                    break
            if not full_sentence:
                full_sentence = sentence_source
            final_sentences.append(full_sentence)
            # Transformer 得分基于长文本分句聚合（如平均值）
            t_score = score_long_text_for_transformer(full_sentence, triple_tuple, aggregation='average')
            transformer_scores.append(t_score)
            # LLM 得分（分三个维度）
            llm_result = score_with_llm(full_sentence, triple_tuple)
            if isinstance(llm_result, dict):
                fa = llm_result.get("factual_accuracy", 0)
                sc = llm_result.get("structural_coherence", 0)
                gr = llm_result.get("groundedness", 0)
            else:
                fa = sc = gr = -1
            llm_fa_scores.append(fa)
            llm_sc_scores.append(sc)
            llm_gr_scores.append(gr)
            print(f"[{i}] Triple: {triple_tuple} | Transformer: {t_score:.2f} | LLM: FA={fa} SC={sc} GR={gr}")
            time.sleep(1)

        # 将最终句子与各评分添加为新列
        triples["sentence"] = final_sentences
        triples["transformer_score"] = transformer_scores
        triples["llm_factual_accuracy"] = llm_fa_scores
        triples["llm_structural_coherence"] = llm_sc_scores
        triples["llm_groundedness"] = llm_gr_scores

        try:
            triples.to_csv(output_path, index=False)
            print(f"已保存评分结果至: {output_path}")
        except Exception as e:
            print(f"保存文件 {output_path} 出现异常: {e}")
            error_logs.append({"file": filename, "error": f"保存异常: {e}"})

    if error_logs:
        error_log_path = os.path.join(output_dir, "error_log.csv")
        pd.DataFrame(error_logs).to_csv(error_log_path, index=False)
        print(f"错误日志已保存至: {error_log_path}")

    print("处理完成！")
