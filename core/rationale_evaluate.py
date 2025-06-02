import json
from typing import List, Dict, Callable, Optional
from transformers import AutoTokenizer
import evaluate


class RationaleEvaluator:
    def __init__(
        self,
        model_name: str,
    ):
        """
        情感推理评估器初始化

        Args:
            model_name: 用于解码的HuggingFace模型路径

        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.meteor_metric = evaluate.load("meteor")
        self.bert_metric = evaluate.load("bertscore")

    @staticmethod
    def format_rationale(rationale: Dict) -> str:
        """将JSON格式的rationale转换为评估用文本"""
        components = [
            rationale["stimulus"]["textual"],
            rationale["stimulus"]["visual"] or "null",
            rationale["appraisal"],
            rationale["response"],
        ]
        return "。".join(comp.strip("。") for comp in components) + "。"

    def decode_output(self, output_ids: List[int]) -> Dict:
        """解码模型输出的token IDs → JSON格式"""
        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        try:
            return json.loads(decoded.strip())
        except json.JSONDecodeError:
            return {
                "stimulus": {"textual": "", "visual": None},
                "appraisal": "",
                "response": "",
            }

    def compute_metrics(
        self,
        predictions: List[List[int]],  # 模型输出的token IDs列表
        references: List[Dict],  # 参考的rationale字典
    ) -> Dict[str, float]:
        """
        计算评估指标

        Returns:
            {
                "meteor": METEOR分数,
                "bert_score": BERTScore F1均值,
                "score_sum": 排名分数总和 (越小越好)
            }
        """
        # 解码预测结果
        pred_dicts = [self.decode_output(ids) for ids in predictions]

        # 格式化为文本
        pred_texts = [self.format_rationale(p) for p in pred_dicts]
        ref_texts = [self.format_rationale(r) for r in references]

        # 计算METEOR
        meteor_result = self.meteor_metric.compute(
            predictions=pred_texts, references=ref_texts
        )
        meteor_score = meteor_result["meteor"]

        # 计算BERTScore
        bert_result = self.bert_metric.compute(
            predictions=pred_texts,
            references=ref_texts,
            lang="zh",
            model_type="bert-base-chinese",
        )
        bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])

        # 计算综合排名分
        meteor_rank = self.rank_fn(meteor_score)
        bert_rank = self.rank_fn(bert_f1)

        return {
            "meteor": round(meteor_score, 4),
            "bert_score": round(bert_f1, 4),
            "score_sum": round(meteor_rank + bert_rank, 4),
        }
