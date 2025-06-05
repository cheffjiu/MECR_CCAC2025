import json
from torch.utils.data import Dataset
from typing import Dict, Any
from feature_loader import PtFeatureLoader, FeatureLoader
from retriever import FAISSRetriever, Retriever
from prompt_constructor import DefaultPromptConstructor, PromptConstructor
from label_constructor import DefaultLabelConstructor, LabelConstructor


class MECRDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        feature_root: str,
        mode: str = None,
        tokenizer: str = None,
        bert_model: str = None,
        # device: str = "cpu",
    ) -> None:
        self.json_path = json_path
        self.feature_root = feature_root
        self.mode = mode
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        # self.device = device
        # 加载json数据
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        # 定义特征加载器
        self.feature_loader = self._create_feature_loader(
            self.feature_root
        )  # FeatureLoader
        self.retriever = self._create_retriever(
            self.tokenizer, self.bert_model
        )  # Retriever
        self.prompt_constructor = self._create_prompt_constructor()  # PromptConstructor
        if self.mode != "test":
            self.label_constructor = (
                self._create_label_constructor()
            )  # LabelConstructor

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 根据idx获取样本
        sample = self.data[idx]

        sid = sample["sample_id"]

        # 加载文本特征和视觉特征
        t_feats, v_feats = self.feature_loader.load_features(sample)
        # 构建样本的FAISS检索库的查询向量
        query_vector = self.retriever.build_query(sample)
        # 从FAISS检索库中检索相似样本
        similar_samples = self.retriever.retrieve(query_vector)

        # 构建LLM提示
        prompt = self.prompt_constructor.build_prompt(sample, similar_samples)

        results = {
            "sample_id": sid,
            "t_feats": t_feats,
            "v_feats": v_feats,
            "utterances": sample["utterances"],
            "prompt": prompt,
        }
        # 构建LLM标签
        if self.mode != "test":
            label = self.label_constructor.build_label_from_sample(sample)
            results["label"] = label
        return results

    def _create_feature_loader(self, feature_root: str) -> "FeatureLoader":
        return PtFeatureLoader(feature_root)

    def _create_retriever(self, tokenizer: str, bert_model: str) -> "Retriever":
        return FAISSRetriever(tokenizer, bert_model)

    def _create_prompt_constructor(self) -> "PromptConstructor":
        return DefaultPromptConstructor()

    def _create_label_constructor(self) -> "LabelConstructor":
        return DefaultLabelConstructor()
