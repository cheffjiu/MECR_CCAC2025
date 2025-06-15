from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Tuple
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import torch
import json
import os

# 设置项目根目录
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), ".."))


class Retriever(ABC):
    @abstractmethod
    def build_query(self, sample: Dict[str, Any]) -> torch.FloatTensor:
        pass

    def retrieve(
        self, query_vector: torch.FloatTensor, k: int = 3
    ) -> List[Dict[str, Any]]:
        pass


class FAISSRetriever(Retriever):
    def __init__(self, tokenizer_instance, bert_model_instance: str) -> None:
        self.tokenizer = tokenizer_instance
        self.bert_model = bert_model_instance

    def build_query(self, sample: Dict[str, Any]) -> torch.FloatTensor:
        utterances = sample["utterances"]
        texts = " ".join([utterance["text"] for utterance in utterances])
        # inputs_ids, attention_mask, token_type_ids 形状：[batch_size, seq_length]->[1,512]
        inputs: Dict[str, torch.FloatTensor] = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        self.bert_model.eval()
        with torch.no_grad():
            # outputs是一个命名元组，包含了last_hidden_state, pooler_output
            # last_hidden_state形状：[batch_size, seq_length, hidden_size]->[1,512,768]
            # pooler_output形状：[batch_size, hidden_size]->[1,768]
            outputs: NamedTuple[torch.FloatTensor] = self.bert_model(**inputs)
        # query_vec形状：[batch_size, hidden_size] -> [1,768]
        query_vec: torch.FloatTensor = outputs.pooler_output.squeeze(0)
        return query_vec

    def retrieve(
        self, query_vector: torch.FloatTensor, k: int = 3
    ) -> List[Dict[str, Any]]:
        # 加载FAISS索引和元数据
        idx_data, meta_data = self._load_faiss_resources()
        # 预处理查询向量
        query_vector = query_vector.to(torch.float32).cpu()
        # 维度校验：将单样本向量 [D] 转换为批处理格式 [1, D]
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)  # 例如 [768] -> [1,768]

        # 内存连续性：确保数组在内存中连续存储（FAISS的硬性要求）
        query_vector = np.ascontiguousarray(
            query_vector, dtype=np.float32
        )  # 强制转换为float32

        # 内存对齐：按列优先(Fortran风格)存储数据（优化FAISS检索速度）
        query_vector = np.require(query_vector, requirements=["F_CONTIGUOUS"])
        # L2归一化
        faiss.normalize_L2(query_vector)
        # FAISS搜索
        # distances: 相似度得分（内积，越大越相似）形状：[batch_size, k] -> [1,3]
        # indices: 对应相似度得分的索引 形状：[batch_size, k] -> [1,3]
        distances, indices = idx_data.search(query_vector, k)
        # 提取top k的索引
        top_k_indices = indices[0].tolist()
        # 根据索引获取对应的元数据
        top_k_rationales: List[Dict[str, Any]] = [
            meta_data[i] for i in top_k_indices if i != -1
        ]  # 过滤无效索引（-1 表示无结果）
        return top_k_rationales

    def _load_faiss_resources(self) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        # 加载索引
        idx_path = os.path.join(project_root, "data/FAISS/faiss_stimulus_index.idx")
        idx_data = faiss.read_index(idx_path)
        meta_path = os.path.join(
            project_root, "data/FAISS/faiss_stimulus_rationale_data.json"
        )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        return idx_data, meta_data
