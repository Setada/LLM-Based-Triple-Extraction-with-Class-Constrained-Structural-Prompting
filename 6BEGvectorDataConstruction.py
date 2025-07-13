# -*- coding: utf-8 -*-
"""
Triple Embedding Generator (emoji‑free)
--------------------------------------
批量读取标注三元组 JSON，按照 `context_text` 字段生成 BGE 嵌入，并输出：

1. embeddings.npy          — (N, 768) NumPy 向量矩阵
2. triples.jsonl           — 每行 id、context_text 及元数据
3. triples.db              — SQLite 小型向量库 (id, vector, metadata)
4. index.faiss             — 可选 FAISS HNSW 索引
5. Milvus / Qdrant         — 可选：上传向量到在线数据库

运行示例：
    python triple_embedding_generator.py \
        --input_file /path/annotated.json \
        --outdir ./embedding_outputs \
        --batch_size 64

参数说明：
    --input_file   标注三元组 JSON 文件路径
    --outdir       输出目录
    --batch_size   推理批大小

可选：
    --push_milvus  将向量写入 Milvus（需本地或远程服务）
    --push_qdrant  将向量写入 Qdrant

"""

import os
import json
import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 可选依赖
try:
    import faiss
except ImportError:
    faiss = None

try:
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
except ImportError:
    Collection = None

try:
    from qdrant_client import QdrantClient, models as qmodels
except ImportError:
    QdrantClient = None

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# ----------------- 工具函数 -----------------

def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_strip(val):
    return val.strip() if isinstance(val, str) else ""

def build_context_text(rec: Dict[str, Any]) -> str:
    desc_subj = safe_strip(rec.get("description_subject", ""))
    desc_obj = safe_strip(rec.get("description_object", ""))
    rel = safe_strip(rec.get("relation", ""))
    event_phase = safe_strip(rec.get("event_phase", ""))
    event_status = safe_strip(rec.get("event_status", ""))
    return (
        f"During {event_phase}, {desc_subj} {rel} {desc_obj}. "
        f"Event status: {event_status}."
    ).strip()

# ----------------- 主流程 -----------------

def main(args):
    # 记录输入目录
    
    INPUT_DIR = r"" # 输入目录

    # 加载模型
    model = SentenceTransformer(MODEL_NAME)

    # 加载数据
    records = load_json_records(args.input_file)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    embeddings = np.zeros((len(records), model.get_sentence_embedding_dimension()), dtype="float32")
    jsonl_path = outdir / "triples.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as jf:
        batch_texts = []
        cursor = 0
        for idx, rec in enumerate(tqdm(records, desc="Embedding")):
            text = build_context_text(rec)
            batch_texts.append(text)
            # 写 JSONL
            jf.write(json.dumps({"id": idx, "context_text": text, **rec}, ensure_ascii=False) + "\n")
            # 批处理推理
            if len(batch_texts) >= args.batch_size or idx == len(records) - 1:
                vecs = model.encode(batch_texts, batch_size=args.batch_size, show_progress_bar=False, normalize_embeddings=True)
                embeddings[cursor: cursor + len(vecs)] = vecs
                cursor += len(vecs)
                batch_texts = []

    np.save(outdir / "embeddings.npy", embeddings)
    print(f"Saved embeddings.npy to {outdir}")

    # SQLite 存储
    db_path = outdir / "triples.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS triples (id INTEGER PRIMARY KEY, vector BLOB, context TEXT)")
    cur.executemany(
        "INSERT OR REPLACE INTO triples (id, vector, context) VALUES (?, ?, ?)",
        [(i, embeddings[i].tobytes(), "") for i in range(len(embeddings))]
    )
    conn.commit(); conn.close()
    print(f"Saved SQLite db to {db_path}")

    # FAISS 索引
    if faiss is not None:
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.add(embeddings)
        faiss.write_index(index, str(outdir / "index.faiss"))
        print("FAISS index saved.")
    else:
        print("faiss 未安装，跳过索引保存。安装： pip install faiss-cpu")

    # 可选：Milvus / Qdrant
    if args.push_milvus and Collection is not None:
        push_to_milvus(embeddings, args)
    if args.push_qdrant and QdrantClient is not None:
        push_to_qdrant(embeddings, args)

# ----------------- 向量数据库函数 -----------------

def push_to_milvus(embeddings: np.ndarray, args):
    connections.connect(host=args.milvus_host, port=args.milvus_port)
    dim = embeddings.shape[1]
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    coll_name = args.milvus_collection
    if coll_name not in Collection.list_collections():
        schema = CollectionSchema(fields)
        Collection(coll_name, schema)
    coll = Collection(coll_name)
    ids = list(range(len(embeddings)))
    coll.insert([ids, embeddings.tolist()])
    print(f"Inserted vectors into Milvus collection '{coll_name}'.")


def push_to_qdrant(embeddings: np.ndarray, args):
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    coll_name = args.qdrant_collection
    dim = embeddings.shape[1]
    if coll_name not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(coll_name, vector_size=dim, distance=qmodels.Distance.COSINE)
    client.upsert(
        collection_name=coll_name,
        points=[qmodels.PointStruct(id=i, vector=embeddings[i].tolist(), payload={}) for i in range(len(embeddings))]
    )
    print(f"Upserted vectors into Qdrant collection '{coll_name}'.")

# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triple Embedding Generator")
    parser.add_argument("--input_file", required=True, help="annotated triples JSON file")
    parser.add_argument("--outdir", default="./embedding_outputs", help="output directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--push_milvus", action="store_true")
    parser.add_argument("--milvus_host", default="localhost")
    parser.add_argument("--milvus_port", default="19530")
    parser.add_argument("--milvus_collection", default="triple_embeddings")
    parser.add_argument("--push_qdrant", action="store_true")
    parser.add_argument("--qdrant_host", default="localhost")
    parser.add_argument("--qdrant_port", default="6333")
    parser.add_argument("--qdrant_collection", default="triple_embeddings")

    main(parser.parse_args())
