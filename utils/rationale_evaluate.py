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
            return self.format_rationale(json.loads(raw_output)) # 这里也尝试解析一下
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
        # --- DEBUG PRINT ---
        print("\n--- DEBUG: In format_rationale ---")
        print(f"Input 'rationale' type: {type(rationale)}")
        print(f"Input 'rationale' content (first 200 chars or full if dict): {str(rationale)[:200] if isinstance(rationale, str) else rationale}")
        # --- END DEBUG ---

        # 核心逻辑：确保 rationale 变量是字典
        if isinstance(rationale, str):
            # 如果 rationale 仍然是一个字符串，说明之前的处理没有成功，
            # 或者它是一个模型生成的原始字符串（非JSON），需要特殊处理。
            # 这里简单返回字符串本身，或者进行更复杂的解析（如safe_extract_rationale）
            # 但为了避免递归循环，我们应该避免再次尝试 json.loads
            # 如果期望它是JSON，那么之前的 json.loads 应该成功了。
            # 如果走到这里是字符串，那它可能就是 plain text。
            print("DEBUG: format_rationale received a string, which is unexpected for direct dictionary access.")
            return self.safe_extract_rationale(rationale) # 尝试用 safe_extract_rationale 处理

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
        
        final_text = "。".join(components) + "。" if components else ""
        # --- DEBUG PRINT ---
        print(f"DEBUG: Formatted rationale output: {final_text[:200]}...")
        print("--- END DEBUG: In format_rationale ---\n")
        # --- END DEBUG ---
        return final_text

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
        # --- DEBUG PRINT ---
        print("\n--- DEBUG: In compute_metrics ---")
        print(f"Input predictions type: {type(predictions)}")
        if len(predictions) > 0:
            print(f"predictions[0] type: {type(predictions[0])}")
            print(f"predictions[0] content: {predictions[0][:200] if isinstance(predictions[0], str) else predictions[0]}")
        print(f"Input references type: {type(references)}")
        if len(references) > 0:
            print(f"references[0] type: {type(references[0])}")
            print(f"references[0] content: {references[0]}")
        # --- END DEBUG ---

        # 预测文本的处理：如果预测是字符串，尝试用 safe_extract_rationale，否则直接用 format_rationale
        pred_texts = [
            self.safe_extract_rationale(p) if isinstance(p, str) else self.format_rationale(p)
            for p in predictions
        ]
        # 参考文本的处理：确保 references 已经是字典列表
        ref_texts = [self.format_rationale(r) for r in references] # 这里的 r 应该是字典了

        # --- DEBUG PRINT ---
        print(f"DEBUG: After processing in compute_metrics:")
        if len(pred_texts) > 0:
            print(f"DEBUG: pred_texts[0] type: {type(pred_texts[0])}, content: {pred_texts[0][:200]}...")
        if len(ref_texts) > 0:
            print(f"DEBUG: ref_texts[0] type: {type(ref_texts[0])}, content: {ref_texts[0][:200]}...")
        print("--- END DEBUG: In compute_metrics ---\n")
        # --- END DEBUG ---

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
