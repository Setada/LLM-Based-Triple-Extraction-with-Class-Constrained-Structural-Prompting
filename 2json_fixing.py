import os
import json
import re

def repair_multi_array_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # 匹配所有数组片段（每个 [] 包裹的对象集合）
    blocks = re.findall(r'\[\s*\{.*?\}\s*\]', raw, flags=re.DOTALL)
    merged = []

    for block in blocks:
        try:
            part = json.loads(block)
            if isinstance(part, list):
                merged.extend(part)
        except Exception as e:
            print(f"跳过无法解析的段落：{e}")

    return merged if merged else None


# 设置路径
src_dir = r" "
dst_dir = r" "
os.makedirs(dst_dir, exist_ok=True)

skipped_files = []

for fname in os.listdir(src_dir):
    if not fname.endswith(".json"):
        continue
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    print(f"正在处理: {fname}")
    try:
        merged_data = repair_multi_array_json(src_path)
        if merged_data:
            with open(dst_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            print(f"修复成功: {fname}")
        else:
            print(f"无合法数据，跳过: {fname}")
            skipped_files.append(fname)
    except Exception as e:
        print(f"发生错误: {fname} | {e}")
        skipped_files.append(fname)

print("\n批处理完成。以下文件未被处理：")
print(skipped_files)
