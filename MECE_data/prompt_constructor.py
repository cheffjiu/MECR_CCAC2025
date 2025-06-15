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
    """
    [重构后]
    一个为小型语言模型优化的Prompt构造器。
    它遵循“模式填充”原则，生成高度结构化、无歧义的Prompt。
    """

    def __init__(self):
        super().__init__()

    def build_prompt(
        self, sample: Dict[str, Any], similar_samples: List[Dict[str, Any]]
    ) -> str:
        """
        构建最终的完整Prompt。
        """
        # 1. 一个极简的、全局性的指令
        instruction = "### 指令:\n根据对话和情感变化，分析其原因。严格遵循输入输出格式。\n\n"

        # 2. 构建Few-shot示例部分
        few_shot_prompt = self._get_prompt_from_similar_samples(similar_samples)

        # 3. 构建需要模型完成的当前任务部分
        task_prompt = self._get_prompt_from_sample(sample)

        return instruction + few_shot_prompt + task_prompt

    def _get_prompt_from_similar_samples(
        self, similar_samples: List[Dict[str, Any]]
    ) -> str:
        """
        根据相似样本构建结构化的 few-shot 示例。
        """
        prompt_parts = []
        for i, item in enumerate(similar_samples, 1):
            # 构建输入部分
            dialogue_history = ""
            for line in item["dialogue"].split("\n"):
                if line.strip():
                    speaker, content = line.split(":", 1)
                    dialogue_history += f"    {speaker.strip()}: {content.strip()}\n"
            
            emo_change = f"从{item['start_emo']}到{item['end_emo']}"

            # 构建输出部分 (!!! 关键改动 !!!)
            # 这里的输出格式必须与我们期望模型生成的格式完全一致
            structured_rationale = self._build_structured_rationale(item)

            # 将示例拼接成一个高度结构化的块
            prompt_parts.append(
                f"### 示例 {i}:\n"
                f"输入:\n"
                f"  对话:\n{dialogue_history.strip()}\n"
                f"  变化: {emo_change}\n"
                f"输出:\n{structured_rationale}\n"
            )

        return "\n".join(prompt_parts)

    def _get_prompt_from_sample(self, sample: Dict[str, Any]) -> str:
        """
        构建最终需要模型推理的任务部分，其结构与示例完全一致。
        """
        # 构建输入部分
        dialogue_history = ""
        for utt in sample["utterances"]:
            # 注意：原始代码情感是列表，这里用 join
            emotion_str = ','.join(utt['emotion'])
            dialogue_history += f"    {utt['speaker']} [{emotion_str}]: {utt['text']}\n"
        
        emo_change = f'从{",".join(sample["emo_change"]["from_emotion"])}到{",".join(sample["emo_change"]["to_emotion"])}'
        
        # 构造最终的任务块，注意末尾的"输出:\n"是引导模型开始生成的关键
        return (
            f"\n### 任务:\n"
            f"输入:\n"
            f"  对话:\n{dialogue_history.strip()}\n"
            f"  变化: {emo_change}\n"
            f"输出:\n"
        )

    def _build_structured_rationale(self, sample: Dict[str, Any]) -> str:
        """
        [!!! 新增辅助函数 !!!]
        根据样本的rationale字典，创建一个结构化的、带标签的字符串。
        这是模型需要学习的输出格式。
        """
        rationale = sample["rationale"]
        textual_stimulus = rationale["stimulus"]["textual"]
        visual_stimulus = rationale["stimulus"]["visual"]
        appraisal = rationale["appraisal"]
        response = rationale["response"]

        # 处理视觉刺激为 None 的情况
        visual_stimulus_str = visual_stimulus if visual_stimulus is not None else "无"

        # 返回与[输出要求]完全一致的多行格式
        return (
            f"[STIMULUS_TEXT]: {textual_stimulus}\n"
            f"[STIMULUS_VISUAL]: {visual_stimulus_str}\n"
            f"[APPRAISAL]: {appraisal}\n"
            f"[RESPONSE]: {response}"
        )

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
#         "sample_id": "anjia_sample1",
#         "dialogue": "A ['Neutral']: 你先把衣服换了，然后去这个地址。\nA ['Neutral']: 一会儿会有装修公司的人过去，你去监工，看看有什么可以帮忙的。\nB ['Sad']: 这个活也派我干呀？\nB ['Sad']: 装修我不懂。\nA ['Anger']: 你发了两天传单了，有没有意向客户啊？要到人家电话没有？",
#         "rationale": {
#             "stimulus": {
#                 "textual": "B表示自己不懂装修",
#                 "visual": None
#             },
#             "appraisal": "A认为B工作态度不好，不但没有为公司带来意向客户，还不想去监工装修",
#             "response": "A感到愤怒"
#         },
#         "start_emo": "Neutral",
#         "end_emo": "Anger"
#     },
#     {
#         "sample_id": "anjia_sample2",
#         "dialogue": "A ['Neutral']: 你先把衣服换了，然后去这个地址。\nA ['Neutral']: 一会儿会有装修公司的人过去，你去监工，看看有什么可以帮忙的。\nB ['Sad']: 这个活也派我干呀？\nB ['Sad']: 装修我不懂。\nA ['Anger']: 你发了两天传单了，有没有意向客户啊？要到人家电话没有？\nB ['Sad', 'Anger']: 你不就光让我发传单吗？没让我要电话呀!",
#         "rationale": {
#             "stimulus": {
#                 "textual": "A对B没能积累意向客户表示不满",
#                 "visual": "A双手环抱胸前瞪着B"
#             },
#             "appraisal": "B认为A只让自己去发传单，没有让自己去获取客户电话",
#             "response": "B感到委屈并且有点生气"
#         },
#         "start_emo": "Sad",
#         "end_emo": "Anger"
#     },
#     {
#         "sample_id": "anjia_sample36",
#         "dialogue": "A ['Neutral']: 其实，我还真舍不得让你去伺候媳妇。\nA ['Neutral']: 有你在店里，能帮我分担不少呢，我这负担也小多了。\nA ['Neutral']: 额，帮我搭把手、干活、说个话，这...\nA ['Happy']: 男女搭配，干活不累嘛，呵呵。\nB ['Sad']: 亲家伺候媳妇，那倒对。\nB ['Sad']: 我也不是她亲妈，不知道她爱吃什么，不爱吃什么。\nB ['Sad']: 要是伺候不好，她不痛快，我也不痛快。\nA ['Happy']: 你这么想就对了，呵呵。\nA ['Happy']: 咱们做父母的，不就是为儿女做贡献的吗，是不是？\nB ['Sad']: 年底咱回去吧，回老家。\nB ['Sad']: 这夏天还能凑合，冬天你那腰根本就不行。\nA ['Surprise']: 我，我回老家？",
#         "rationale": {
#             "stimulus": {
#                 "textual": "B表示要和A年底回老家",
#                 "visual": None
#             },
#             "appraisal": "A没想到B会突然提出要回老家，这让他感到意外",
#             "response": "A有点惊讶"
#         },
#         "start_emo": "Happy",
#         "end_emo": "Surprise"
#     },
# ]
# # 生成最终结果
# final_result = sample_prompt.build_prompt(sample, similary_samples)
# print(final_result)
