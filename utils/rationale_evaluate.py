import json
from typing import List, Dict, Any
from transformers import AutoTokenizer
import evaluate
import nltk
import re



class RationaleEvaluator:
    def __init__(self, model_name: str):
        """
        情感归因评估器初始化。

        Args:
            model_path: 用于解码的 HuggingFace 模型路径（主要用于获取 tokenizer）。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.meteor_metric = evaluate.load("meteor")
        self.bert_metric = evaluate.load("bertscore")
        # BERTScore 的模型类型在 compute_metrics 中指定为 "bert-base-chinese"

    def decode_generated_tokens(self, generated_ids: List[List[int]]) -> List[str]:
        """
        将模型生成的 token ID 解码成带标签的多行纯文本。

        Args:
            generated_ids: 模型 generate 方法输出的 token ID 列表，
                           其中每个子列表代表一个生成序列的 token IDs。

        Returns:
            List[str]: 解码后的带标签的多行纯文本字符串列表。
        """
        # 注意：这里假设 generated_ids 已经经过了切片处理，即移除了 prompt 部分
        decoded_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return decoded_texts

    def _parse_tagged_text_to_dict(self, tagged_text: str) -> Dict[str, Any]:
        """
        内部方法：将带标签的多行纯文本解析成一个中间字典。
        这是 LLM 实际输出的格式（或监督 Label 的格式）。
        """
        parsed_data = {
            "stimulus_textual": None,
            "stimulus_visual": None,
            "appraisal": None,
            "response": None,
        }

        if not tagged_text or len(tagged_text.strip()) == 0:
            return parsed_data

        lines = tagged_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("[STIMULUS_TEXT]:"):
                content = line.replace("[STIMULUS_TEXT]:", "", 1).strip()
                parsed_data["stimulus_textual"] = content
            elif line.startswith("[STIMULUS_VISUAL]:"):
                content = line.replace("[STIMULUS_VISUAL]:", "", 1).strip()
                parsed_data["stimulus_visual"] = (
                    content if content.lower() != "无" else None
                )
            elif line.startswith("[APPRAISAL]:"):
                content = line.replace("[APPRAISAL]:", "", 1).strip()
                parsed_data["appraisal"] = content
            elif line.startswith("[RESPONSE]:"):
                content = line.replace("[RESPONSE]:", "", 1).strip()
                parsed_data["response"] = content
        return parsed_data

    def _format_dict_to_eval_text(self, data_dict: Dict[str, Any]) -> str:
        """
        内部方法：将解析后的中间字典（或原始JSON结构）转换为最终评估所需的单行纯文本格式。
        即："{textual stimulus}；{visual stimulus}。{appraisal}。{response}。"
        """
        textual = data_dict.get("stimulus_textual")
        visual = data_dict.get("stimulus_visual")
        appraisal = data_dict.get("appraisal")
        response = data_dict.get("response")

        stimulus_parts = []
        if textual:
            # 移除末尾句号，以免与后续连接的句号重复
            stimulus_parts.append(textual.strip("。"))

        if visual and visual.strip().lower() != "null":
            if stimulus_parts:  # 如果文本刺激存在，则用分号连接
                stimulus_parts[0] += f"；{visual.strip('。')}"
            else:  # 如果文本刺激不存在，则直接添加视觉刺激
                stimulus_parts.append(visual.strip("。"))

        formatted_stimulus_part = "".join(stimulus_parts)

        components = []
        if formatted_stimulus_part:
            components.append(formatted_stimulus_part)

        if appraisal:
            components.append(appraisal.strip("。"))

        if response:
            components.append(response.strip("。"))

        # 使用句号连接所有主要部分，并在末尾加上一个句号
        final_text = "。".join(components) + "。" if components else ""
        return final_text

    def prepare_for_evaluation(self, input_data: Any) -> str:
        """
        将任何有效输入（LLM 原始输出字符串、原始 JSON 字典、或解析后的中间字典）
        统一转换为评估所需的最终单行纯文本格式。
        """
        if isinstance(input_data, str):
            # 如果是 LLM 原始输出（带标签的纯文本），先解析成字典
            parsed_dict = self._parse_tagged_text_to_dict(input_data)
            return self._format_dict_to_eval_text(parsed_dict)
        elif isinstance(input_data, dict):
            # 如果是原始 JSON 字典，需要从中提取出 rationale 部分，并将其转换为中间字典格式，
            # 或者直接从原始 JSON 中提取字段进行格式化。
            # 为了简化，我们统一转换为中间字典结构，再进行格式化。
            rationale_core = input_data.get(
                "rationale", input_data
            )  # 兼容直接是rationale字典或包含rationale的字典

            # 这里的字段名与 _parse_tagged_text_to_dict 的输出字段名保持一致
            intermediate_dict = {
                "stimulus_textual": rationale_core.get("stimulus", {}).get("textual"),
                "stimulus_visual": rationale_core.get("stimulus", {}).get("visual"),
                "appraisal": rationale_core.get("appraisal"),
                "response": rationale_core.get("response"),
            }
            # 特殊处理 visual 为 None 的情况，使其与 _parse_tagged_text_to_dict 的 "无" 保持一致，再传给 _format_dict_to_eval_text
            if intermediate_dict["stimulus_visual"] is None:
                intermediate_dict["stimulus_visual"] = (
                    "无"  # 临时的，只为 _format_dict_to_eval_text 处理
                )

            return self._format_dict_to_eval_text(intermediate_dict)
        else:
            raise TypeError(
                f"Unsupported input type for preparation: {type(input_data)}"
            )

    def compute_metrics(
        self,
        predictions: List[str],  # 预测是LLM直接输出的带标签多行纯文本字符串列表
        references: List[
            str
        ],  # 参考是原始 JSON 格式的 rationale 字典列表 (来自您的数据加载器)
    ) -> Dict[str, float]:
        """
        计算评估指标。

        Args:
            predictions: LLM 直接输出的**带标签多行纯文本**字符串列表。
            references: 原始 JSON 格式的 rationale 字典列表 (来自数据加载器)。

        Returns:
            Dict[str, float]: 包含 meteor, bert_score 和 score_sum 的字典。
        """
        # 1. 将预测和参考都转换为评估所需的最终单行纯文本格式
        # predictions 是 LLM 的带标签输出字符串，需要先解析再格式化
        pred_texts = predictions

        # references 是原始 JSON 字典，也需要通过 prepare_for_evaluation 格式化
        ref_texts = [self.prepare_for_evaluation(r_dict) for r_dict in references]
        
        # 确保预测和参考数量一致
        if len(pred_texts) != len(ref_texts):
            print(
                f"WARNING: Mismatched lengths of predictions ({len(pred_texts)}) and references ({len(ref_texts)}). Truncating to minimum length."
            )
            min_len = min(len(pred_texts), len(ref_texts))
            pred_texts = pred_texts[:min_len]
            ref_texts = ref_texts[:min_len]

        # 计算 METEOR 分数
        meteor_result = self.meteor_metric.compute(
            predictions=pred_texts, references=ref_texts
        )
        meteor_score = meteor_result.get("meteor", 0.0)

        # 计算 BERTScore 分数
        bert_result = self.bert_metric.compute(
            predictions=pred_texts,
            references=ref_texts,
            lang="zh",
            model_type="bert-base-chinese",  # 指定中文 BERT 模型
        )
        bert_f1 = (
            sum(bert_result["f1"]) / len(bert_result["f1"])
            if bert_result and "f1" in bert_result and len(bert_result["f1"]) > 0
            else 0.0
        )

        # score_sum 越大越好
        return {
            "meteor": round(meteor_score, 4),
            "bert_score": round(bert_f1, 4),
            "score_sum": round(meteor_score + bert_f1, 4),
        }
