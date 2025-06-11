from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Tuple
import json


class PromptConstructor(ABC):
    @abstractmethod
    def build_prompt(
        self, sample: Dict[str, Any], similar_samples: List[Dict[str, Any]]
    ) -> str:
        pass


class DefaultPromptConstructor(PromptConstructor):
    def __init__(self):
        super().__init__()

    def build_prompt(
        self, sample: Dict[str, Any], similar_samples: List[Dict[str, Any]]
    ) -> str:
        prompt = "任务角色："
        prompt += "请根据对话历史和[情感变化]内容的提示，给出完整的情感归因，情感归因内容包含文本刺激、视觉刺激（如果没有视觉线索就填写“无”）、评估推理和情绪反应。\n"
        prompt_similar_samples = self._get_prompt_from_similar_samples(similar_samples)
        prompt += prompt_similar_samples
        prompt_sample = self._get_prompt_from_sample(sample)
        prompt += prompt_sample
        return prompt

    def _get_prompt_from_similar_samples(
        self, similar_samples: List[Dict[str, Any]]
    ) -> str:
        prompt_similar_samples = ""
        for i, item in enumerate(similar_samples, 1):
            # 构建对话历史部分
            dialogue_history = "对话历史：\n"
            for line in item["dialogue"].split("\n"):
                if line.strip():  # 跳过空行
                    speaker, content = line.split(":", 1)
                    dialogue_history += f"{speaker}: {content.strip()}\n"
            # 情感变化部分

            emo_change = f"[情感变化]:从{item['start_emo']}到{item['end_emo']}\n"
            # 构建情感归因部分
            rationale = self._build_rationale_from_sample(item)
            # 拼接完整格式
            prompt_similar_samples += f"示例样本{i}：\n{dialogue_history}{emo_change}\n情感归因：\n{rationale}\n"

        return prompt_similar_samples

    def _get_prompt_from_sample(self, sample: Dict[str, Any]) -> str:
        # 构建对话历史部分
        dialogue_history = ""
        for utt in sample["utterances"]:
            dialogue_history += (
                f"{utt['speaker']} [{','.join(utt['emotion'])}]: {utt['text']}\n"
            )
        emo_change = f'[情感变化]:从{",".join(sample["emo_change"]["from_emotion"])}到{",".join(sample["emo_change"]["to_emotion"])}\n'  # 修改 [输出要求] 部分，明确指定纯文本格式
        output_requirements = """[输出要求]
        严格按照以下带有标签的格式输出，每部分独占一行，不要添加任何解释性语言或额外说明：
        [STIMULUS_TEXT]: {填写文本刺激}
        [STIMULUS_VISUAL]: {填写视觉线索，若无则填写“无”}
        [APPRAISAL]: {填写评估推理}
        [RESPONSE]: {填写情绪反应}"""

        return f"\n[当前任务]\n请根据[情感变化]内容的提示和对话历史，按照上述格式生成完整的情感归因内容。\n对话历史：\n{dialogue_history}{emo_change}"

    def _build_rationale_from_sample(self, sample: Dict[str, Any]) -> str:
        rationale = sample["rationale"]
        textual_stimulus = rationale["stimulus"]["textual"]
        visual_stimulus = rationale["stimulus"]["visual"]
        appraisal = rationale["appraisal"]
        response = rationale["response"]

        # 处理视觉刺激为 None 的情况，统一表示为 "无"
        visual_stimulus_str = visual_stimulus if visual_stimulus is not None else "无"

        label_parts = []
        label_parts.append(f"[STIMULUS_TEXT]: {textual_stimulus}")
        label_parts.append(f"[STIMULUS_VISUAL]: {visual_stimulus_str}")
        label_parts.append(f"[APPRAISAL]: {appraisal}")
        label_parts.append(f"[RESPONSE]: {response}")

        # 使用 '\n' 将所有部分连接起来，形成多行字符串
        return " ".join(label_parts)


# sample_prompt = DefaultPromptConstructor()
# # 示例数据
# sample = {
#     "sample_id": "anjia_sample37",
#     "video_path": "anjia/anjia_11.mp4",
#     "utterances": [
#         {
#             "utt_id": "anjia_11_1",
#             "text": "其实，我还真舍不得让你去伺候媳妇。",
#             "speaker": "A",
#             "start_sec": 0.04,
#             "end_sec": 4.08,
#             "emotion": ["Neutral"],
#         },
#         {
#             "utt_id": "anjia_11_2",
#             "text": "有你在店里，能帮我分担不少呢，我这负担也小多了。",
#             "speaker": "A",
#             "start_sec": 4.08,
#             "end_sec": 10.4,
#             "emotion": ["Neutral"],
#         },
#         {
#             "utt_id": "anjia_11_3",
#             "text": "额，帮我搭把手、干活、说个话，这...",
#             "speaker": "A",
#             "start_sec": 10.4,
#             "end_sec": 16.68,
#             "emotion": ["Neutral"],
#         },
#         {
#             "utt_id": "anjia_11_4",
#             "text": "男女搭配，干活不累嘛，呵呵。",
#             "speaker": "A",
#             "start_sec": 16.68,
#             "end_sec": 18.6,
#             "emotion": ["Happy"],
#         },
#         {
#             "utt_id": "anjia_11_5",
#             "text": "亲家伺候媳妇，那倒对。",
#             "speaker": "B",
#             "start_sec": 18.6,
#             "end_sec": 30.32,
#             "emotion": ["Sad"],
#         },
#         {
#             "utt_id": "anjia_11_6",
#             "text": "我也不是她亲妈，不知道她爱吃什么，不爱吃什么。",
#             "speaker": "B",
#             "start_sec": 30.32,
#             "end_sec": 34.92,
#             "emotion": ["Sad"],
#         },
#         {
#             "utt_id": "anjia_11_7",
#             "text": "要是伺候不好，她不痛快，我也不痛快。",
#             "speaker": "B",
#             "start_sec": 34.92,
#             "end_sec": 38.72,
#             "emotion": ["Sad"],
#         },
#         {
#             "utt_id": "anjia_11_8",
#             "text": "你这么想就对了，呵呵。",
#             "speaker": "A",
#             "start_sec": 38.72,
#             "end_sec": 41.32,
#             "emotion": ["Happy"],
#         },
#         {
#             "utt_id": "anjia_11_9",
#             "text": "咱们做父母的，不就是为儿女做贡献的吗，是不是？",
#             "speaker": "A",
#             "start_sec": 41.32,
#             "end_sec": 46.76,
#             "emotion": ["Happy"],
#         },
#         {
#             "utt_id": "anjia_11_10",
#             "text": "年底咱回去吧，回老家。",
#             "speaker": "B",
#             "start_sec": 46.76,
#             "end_sec": 61.4,
#             "emotion": ["Sad"],
#         },
#         {
#             "utt_id": "anjia_11_11",
#             "text": "这夏天还能凑合，冬天你那腰根本就不行。",
#             "speaker": "B",
#             "start_sec": 61.4,
#             "end_sec": 65.68,
#             "emotion": ["Sad"],
#         },
#         {
#             "utt_id": "anjia_11_12",
#             "text": "我，我回老家？",
#             "speaker": "A",
#             "start_sec": 65.68,
#             "end_sec": 67.48,
#             "emotion": ["Surprise"],
#         },
#         {
#             "utt_id": "anjia_11_13",
#             "text": "那哪成啊！那，那我还能天天看见我孙子吗？是吧。",
#             "speaker": "A",
#             "start_sec": 67.48,
#             "end_sec": 72.36,
#             "emotion": ["Anger", "Sad"],
#         },
#     ],
#     "key_frames": [
#         "anjia_11_1_frame_26.jpg",
#         "anjia_11_1_frame_51.jpg",
#         "anjia_11_1_frame_76.jpg",
#         "anjia_11_2_frame_141.jpg",
#         "anjia_11_2_frame_181.jpg",
#         "anjia_11_2_frame_220.jpg",
#         "anjia_11_3_frame_299.jpg",
#         "anjia_11_3_frame_338.jpg",
#         "anjia_11_3_frame_377.jpg",
#         "anjia_11_4_frame_429.jpg",
#         "anjia_11_4_frame_441.jpg",
#         "anjia_11_4_frame_453.jpg",
#         "anjia_11_5_frame_538.jpg",
#         "anjia_11_5_frame_611.jpg",
#         "anjia_11_5_frame_684.jpg",
#         "anjia_11_6_frame_786.jpg",
#         "anjia_11_6_frame_815.jpg",
#         "anjia_11_6_frame_844.jpg",
#         "anjia_11_7_frame_896.jpg",
#         "anjia_11_7_frame_920.jpg",
#         "anjia_11_7_frame_944.jpg",
#         "anjia_11_8_frame_984.jpg",
#         "anjia_11_8_frame_1000.jpg",
#         "anjia_11_8_frame_1016.jpg",
#         "anjia_11_9_frame_1067.jpg",
#         "anjia_11_9_frame_1101.jpg",
#         "anjia_11_9_frame_1135.jpg",
#         "anjia_11_10_frame_1260.jpg",
#         "anjia_11_10_frame_1352.jpg",
#         "anjia_11_10_frame_1443.jpg",
#         "anjia_11_11_frame_1561.jpg",
#         "anjia_11_11_frame_1588.jpg",
#         "anjia_11_11_frame_1615.jpg",
#         "anjia_11_12_frame_1653.jpg",
#         "anjia_11_12_frame_1664.jpg",
#         "anjia_11_12_frame_1675.jpg",
#         "anjia_11_13_frame_1717.jpg",
#         "anjia_11_13_frame_1748.jpg",
#         "anjia_11_13_frame_1778.jpg",
#     ],
#     "emo_change": {
#         "start_idx": 11,
#         "end_idx": 12,
#         "from_emotion": ["Surprise"],
#         "to_emotion": ["Anger", "Sad"],
#     },
#     "rationale": {
#         "stimulus": {"textual": "B让A和自己年底回老家", "visual": None},
#         "appraisal": "A舍不得离开孙子，不想回老家",
#         "response": "A有些着急，也有点失落",
#     },
# }
# similary_samples = [
#     {
#         "sample_id": "anjia_sample2",
#         "dialogue": "A ['Neutral']: 你先把衣服换了，然后去这个地址。\nA ['Neutral']: 一会儿会有装修公司的人过去，你去监工，看看有什么可以帮忙的。\nB ['Sad']: 这个活也派我干呀？\nB ['Sad']: 装修我不懂。\nA ['Anger']: 你发了两天传单了，有没有意向客户啊？要到人家电话没有？\nB ['Sad', 'Anger']: 你不就光让我发传单吗？没让我要电话呀!",
#         "rationale": {
#             "stimulus": {
#                 "textual": "A对B没能积累意向客户表示不满",
#                 "visual": "A双手环抱胸前瞪着B",
#             },
#             "appraisal": "B认为A只让自己去发传单，没有让自己去获取客户电话",
#             "response": "B感到委屈并且有点生气",
#         },
#         "start_emo": "Sad",
#         "end_emo": "Anger",
#     },
#     # {
#     #     "dialogue": "A [Neutral]: 你先把衣服换了，然后去这个地址。\nA [Neutral]: 一会儿会有装修公司的人过去，你去监工，看看有什么可以帮忙的。\nB [Sad]: 这个活也派我干呀？\nB [Sad]: 装修我不懂。\nA [Anger]: 你发了两天传单了，有没有意向客户啊？要到人家电话没有？\nB [Sad]: 你不就光让我发传单吗？没让我要电话呀!",
#     #     "rationale": {
#     #         "stimulus": {
#     #             "textual": "A对B没能积累意向客户表示不满",
#     #             "visual": "A双手环抱胸前瞪着B",
#     #         },
#     #         "appraisal": "B认为A只让自己去发传单，没有让自己去获取客户电话",
#     #         "response": "B感到委屈并且有点生气",
#     #     },
#     # },
#     # {
#     #     "dialogue": "A [Neutral]: 其实，我还真舍不得让你去伺候媳妇。\nA [Neutral]: 有你在店里，能帮我分担不少呢，我这负担也小多了。\nA [Neutral]: 额，帮我搭把手、干活、说个话，这...\nA [Happy]: 男女搭配，干活不累嘛，呵呵。\nB [Sad]: 亲家伺候媳妇，那倒对。\nB [Sad]: 我也不是她亲妈，不知道她爱吃什么，不爱吃什么。\nB [Sad]: 要是伺候不好，她不痛快，我也不痛快。\nA [Happy]: 你这么想就对了，呵呵。\nA [Happy]: 咱们做父母的，不就是为儿女做贡献的吗，是不是？\nB [Sad]: 年底咱回去吧，回老家。\nB [Sad]: 这夏天还能凑合，冬天你那腰根本就不行。\nA [Surprise]: 我，我回老家？",
#     #     "rationale": {
#     #         "stimulus": {"textual": "B表示要和A年底回老家", "visual": None},
#     #         "appraisal": "A没想到B会突然提出要回老家，这让他感到意外",
#     #         "response": "A有点惊讶",
#     #     },
#     # },
# ]
# # 生成最终结果
# final_result = sample_prompt.build_prompt(sample, similary_samples)
# print(final_result)
