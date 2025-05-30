import faiss
import numpy as np
import json
import os
import torch

from build_prompt import build_prompt_from_retrieval, format_rationale

# 获取项目根目录（与 faiss_build.py 保持一致）
current_file_path = os.path.abspath(__file__)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), ".."))


def load_faiss_resources():
    """加载 FAISS 索引和元数据"""
    # 加载索引
    idx_path = os.path.join(workspace_root, "data/FAISS", "faiss_stimulus_index.idx")
    index = faiss.read_index(idx_path)

    # 加载元数据（rationale 字典列表）
    rationale_path = os.path.join(
        workspace_root, "data/FAISS", "faiss_stimulus_rationale_data.json"
    )
    with open(rationale_path, "r", encoding="utf-8") as f:
        rationale_data = json.load(f)

    return index, rationale_data


def query_similar_rationale(q, k=5):
    """
    查询前k个最相似的 rationale 数据
    Args:
        q (np.ndarray): 查询特征向量（需与索引维度一致，shape=(D,) 或 (1, D)）
        k (int): 前k个相似结果
    Returns:
        list: 前k个相似的 rationale 字典列表（按相似度从高到低排序）
    """
    # 加载资源
    index, rationale_data = load_faiss_resources()

    # 预处理查询特征（增加维度校验和显式F顺序转换）
    q = q.to(torch.float32).cpu()
    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = np.ascontiguousarray(q, dtype=np.float32)  # 显式指定dtype
    q = np.require(q, requirements=["F_CONTIGUOUS"])  # 强制F顺序

    # L2 归一化
    faiss.normalize_L2(q)

    # FAISS 搜索
    distances, indices = index.search(q, k)  # distances: 相似度得分（内积，越大越相似）
    top_k_indices = indices[0].tolist()  # 提取 top k 的索引（假设 batch_size=1）

    # 根据索引获取对应的 rationale 数据
    top_k_rationales = [
        rationale_data[i] for i in top_k_indices if i != -1
    ]  # 过滤无效索引（-1 表示无结果）

    return top_k_rationales
