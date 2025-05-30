# fusion_feature_dataset.py

import os
import json
import torch
from torch.utils.data import Dataset

# 导入 FAISS 相关模块和 Prompt 构建函数
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "FAISS_database"))

from FAISS_database.build_query import process_utterances_for_query
from FAISS_database.faiss_query import query_similar_rationale, load_faiss_resources
from FAISS_database.build_prompt import (
    build_prompt_from_retrieval,
    concatenate_utterances,
    format_rationale,
)


class FusionFeatureDataset(Dataset):

    def __init__(
        self,
        feature_root=None,
        json_path=None,
        mode="train",
        tokenizer=None,
        bert_model=None,
        device="cpu",
    ):
        self.feature_root = feature_root
        self.mode = mode
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device

        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if self.mode != "test":
            if self.tokenizer is None or self.bert_model is None:
                raise ValueError(
                    "In 'train' or 'val' mode, tokenizer and bert_model must be provided for FAISS query and prompt building."
                )
            print("正在加载 FAISS 索引和元数据...")
            self.faiss_index, self.faiss_metadata = load_faiss_resources()
            self.bert_model.eval()
            for param in self.bert_model.parameters():
                param.requires_grad = False
            print("FAISS 资源加载完成。")
        else:
            self.faiss_index = None
            self.faiss_metadata = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sid = sample["sample_id"]

        try:
            text_path = os.path.join(
                self.feature_root, sid, "text_feature", f"{sid}.pt"
            )
            vis_path = os.path.join(
                self.feature_root, sid, "video_feature", f"{sid}_video.pt"
            )
            t_feats = torch.load(text_path)
            v_all = torch.load(vis_path)
        except Exception as e:
            raise RuntimeError(f"加载特征失败 - 样本ID: {sid}\n错误详情: {str(e)}")

        if len(t_feats) != v_all.shape[0]:
            raise ValueError(
                f"特征时序长度不一致 - 样本ID: {sid}\n"
                f"文本特征长度: {len(t_feats)}\n"
                f"视觉特征长度: {v_all.shape[0]}"
            )

        rationale_llm_data = None
        if self.mode != "test":
            rationale_llm_data = self._build_llm_prompt_and_label(sample)

        return {
            "sample_id": sid,
            "t_feats": t_feats,
            "v_feats": v_all[:, 1:],
            "v_mask": v_all[:, 0].bool(),
            "utterances": sample["utterances"],
            "change_span": (
                sample["emo_change"]["start_idx"],
                sample["emo_change"]["end_idx"],
            ),
            "rationale_llm_data": rationale_llm_data,
        }

    def _build_llm_prompt_and_label(self, sample):
        """
        根据样本数据和检索结果构建 LLM 的 Prompt 和 Label
        """
        emo_change = sample["emo_change"]
        start_idx = emo_change["start_idx"]
        end_idx = emo_change["end_idx"]

        # 确保传递给 process_utterances_for_query 的 utterances 是字符串列表
        utterances = [
            utt.get("text", "") for utt in sample["utterances"] if isinstance(utt, dict)
        ]

        query_vec = process_utterances_for_query(
            utterances=utterances,
            tokenizer=self.tokenizer,
            bert_model=self.bert_model,
            device=self.device,
        )

        retrieved_examples = query_similar_rationale(
            q=query_vec,
            k=3,
        )

        prompt_text = build_prompt_from_retrieval(
            retrieved_examples=retrieved_examples,
            current_sample=sample,
            include_prefix=True,
        )

        r = sample["rationale"]
        label_components = ["Rationale:"]

        stimulus_textual = r["stimulus"].get("textual")
        stimulus_visual = r["stimulus"].get("visual")
        appraisal = r.get("appraisal")
        response = r.get("response")

        if stimulus_textual:
            label_components.append(f"Stimulus textual: {stimulus_textual}")
        else:
            label_components.append("Stimulus textual:")

        if stimulus_visual:
            label_components.append(f"Stimulus visual: {stimulus_visual}")
        else:
            label_components.append("Stimulus visual:")

        if appraisal:
            label_components.append(f"Appraisal: {appraisal}")
        else:
            label_components.append("Appraisal:")

        if response:
            label_components.append(f"Response: {response}")
        else:
            label_components.append("Response:")

        label_text = "\n".join(label_components)

        return {"prompt_text": prompt_text, "label_text": label_text}
