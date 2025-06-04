import json
from typing import List, Dict
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
        """
        将JSON格式的rationale转换为评估用文本
        Args:
            rationale (Dict): 包含情感归因信息的字典。
                              预期结构为 {"rationale": {"stimulus": {...}, "appraisal": "...", "response": "..."}}
                              或者直接是 {"stimulus": {...}, "appraisal": "...", "response": "..."}
        """
        if "rationale" in rationale and isinstance(rationale["rationale"], dict):
            actual_rationale = rationale["rationale"]
        else:
            actual_rationale = rationale

        stimulus_parts = []
        if "stimulus" in actual_rationale and actual_rationale["stimulus"]:
            if (
                "textual" in actual_rationale["stimulus"]
                and actual_rationale["stimulus"]["textual"]
            ):
                stimulus_parts.append(
                    actual_rationale["stimulus"]["textual"].strip("。")
                )

            # Only append visual if it exists and is not null
            if (
                "visual" in actual_rationale["stimulus"]
                and actual_rationale["stimulus"]["visual"]
                and actual_rationale["stimulus"]["visual"].strip() != "null"
            ):
                stimulus_parts.append(
                    actual_rationale["stimulus"]["visual"].strip("。")
                )

        formatted_stimulus = ";".join(stimulus_parts)

        components = []
        if formatted_stimulus:
            components.append(formatted_stimulus)

        if "appraisal" in actual_rationale and actual_rationale["appraisal"]:
            components.append(actual_rationale["appraisal"].strip("。"))

        if "response" in actual_rationale and actual_rationale["response"]:
            components.append(actual_rationale["response"].strip("。"))

        return "。".join(components) + "。"

    def compute_metrics(
        self,
        predictions: List[Dict],  # 现在直接接收已解码的JSON字典列表
        references: List[Dict],  # 参考的rationale字典
    ) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            predictions (List[Dict]): 模型生成的已解码JSON字典列表。
            references (List[Dict]): 真实的标签rationale字典列表。

        Returns:
            {
                "meteor": METEOR分数,
                "bert_score": BERTScore F1均值,
                "score_sum": 排名分数总和 (越小越好)
            }
        """
        # 预测结果已经是JSON字典，直接格式化为文本
        pred_texts = [self.format_rationale(p) for p in predictions]
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

        return {
            "meteor": round(meteor_score, 4),
            "bert_score": round(bert_f1, 4),
            "score_sum": round(meteor_score + bert_f1, 4),
        }


# test_rationale = RationaleEvaluator(model_name="Qwen/Qwen3-0.6B")
# data = {
#     "rationale": {
#         "stimulus": {
#             "textual": "A对B没能积累意向客户表示不满",
#             "visual": "A的表情变化：愤怒 -> 惊讶 -> 开心",
#         },
#         "appraisal": "B认为A只让自己去发传单，没有让自己去获取客户电话",
#         "response": "B感到委屈并且有点生气",
#     }
# }
# result = test_rationale.format_rationale(data["rationale"])
# print(result)
