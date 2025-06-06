import json
from typing import List, Dict
from transformers import AutoTokenizer
import evaluate
import nltk
import re



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

    def safe_extract_rationale(self, raw_output: str):
        """
        尽力从模型输出中提取 rationale 字段中的 stimulus, appraisal, response 文本。
        即使原始输出不是合法 JSON。
        """
        if not raw_output or len(raw_output.strip()) == 0:
            return ""  # 空字符串

        # 尝试直接加载
        try:
            return self.format_rationale(json.loads(raw_output))
        except Exception:
            pass  # json.loads 失败就尝试正则提取

        # 使用正则匹配三段内容（兼容中文引号、缩进等格式）
        stimulus_textual_match = re.search(r'"textual"\s*:\s*["“](.*?)["”]', raw_output)
        stimulus_visual_match = re.search(r'"visual"\s*:\s*["“](.*?)["”]', raw_output)
        appraisal_match = re.search(r'"appraisal"\s*:\s*["“](.*?)["”]', raw_output)
        response_match = re.search(r'"response"\s*:\s*["“](.*?)["”]', raw_output)

        parts = []
        if stimulus_textual_match:
            parts.append(stimulus_textual_match.group(1).strip("。"))
        if stimulus_visual_match and stimulus_visual_match.group(1).strip().lower() != "null":
            parts.append(stimulus_visual_match.group(1).strip("。"))
        if appraisal_match:
            parts.append(appraisal_match.group(1).strip("。"))
        if response_match:
            parts.append(response_match.group(1).strip("。"))

        return "。".join(parts) + "。" if parts else ""

    def format_rationale(self, rationale: Dict) -> str:
        """
        将JSON格式的rationale转换为评估用文本
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

            if (
                "visual" in actual_rationale["stimulus"]
                and actual_rationale["stimulus"]["visual"]
                and actual_rationale["stimulus"]["visual"].strip().lower() != "null"
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

        return "。".join(components) + "。" if components else ""

    def compute_metrics(
        self,
        predictions: List[Dict],  # 现在直接接收已解码的JSON字典列表或字符串输出
        references: List[Dict],   # 参考的rationale字典
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
        pred_texts = [
            self.safe_extract_rationale(p) if isinstance(p, str) else self.format_rationale(p)
            for p in predictions
        ]
        ref_texts = [self.format_rationale(r) for r in references]

        meteor_result = self.meteor_metric.compute(
            predictions=pred_texts, references=ref_texts
        )
        meteor_score = meteor_result["meteor"]

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
