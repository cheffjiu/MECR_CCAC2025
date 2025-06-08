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
        prompt = "[任务角色]\n"
        prompt += "你是一个多模态对话情感归因分析专家。请根据输入的对话历史，判断说话人的情绪变化，并给出完整的情感归因逻辑。\n\n"
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

            # 构建情感归因部分
            rationale = self._build_rationale_from_sample(item)      
            # 拼接完整格式
            prompt_similar_samples += f'[示例样本{i}]\n{dialogue_history}\n情感归因：\n{rationale}\n\n'

        return prompt_similar_samples

    def _get_prompt_from_sample(self, sample: Dict[str, Any]) -> str:
        # 构建对话历史部分
        dialogue_history = ""
        for utt in sample["utterances"]:
            dialogue_history += (
                f"{utt['speaker']} [{','.join(utt['emotion'])}]: {utt['text']}\n"
            )

        # 修改 [输出要求] 部分，明确指定纯文本格式
        output_requirements = """[输出要求]
- 请从对话中识别**触发情绪变化的刺激**（Stimulus），包括文本/视觉；
- 解释说话人的**评估与推理过程**（Appraisal）；
- 明确指出**最终情绪反应**（Response）；
- **严格按照以下带有标签的格式输出**，每部分独占一行，不要添加任何解释性语言或额外说明：
  - **文本刺激：** `[STIMULUS_TEXT]: {填写文本刺激}`
  - **视觉刺激：** `[STIMULUS_VISUAL]: {填写视觉线索，若无则填写“无”}`
  - **评估推理：** `[APPRAISAL]: {填写评估推理}`
  - **情绪反应：** `[RESPONSE]: {填写情绪反应}`"""

        return f'[当前任务]\n请根据以下新对话，仿照上述格式输出完整的情感归因。\n\n对话历史：\n{dialogue_history}\n\n{output_requirements}\n'
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
        return "\n".join(label_parts)

sample_prompt = DefaultPromptConstructor()
# 示例数据
sample = {
    "sample_id": "anjia_sample1",
    "video_path": "anjia/anjia_1.mp4",
    "utterances": [
        {
            "utt_id": "anjia_1_1",
            "text": "你先把衣服换了，然后去这个地址。",
            "speaker": "A",
            "start_sec": 0.04,
            "end_sec": 2.76,
            "emotion": ["Neutral"],
        },
        {
            "utt_id": "anjia_1_2",
            "text": "一会儿会有装修公司的人过去，你去监工，看看有什么可以帮忙的。",
            "speaker": "A",
            "start_sec": 2.76,
            "end_sec": 7.8,
            "emotion": ["Neutral"],
        },
        {
            "utt_id": "anjia_1_3",
            "text": "这个活也派我干呀？",
            "speaker": "B",
            "start_sec": 7.8,
            "end_sec": 12.48,
            "emotion": ["Sad"],
        },
        {
            "utt_id": "anjia_1_4",
            "text": "装修我不懂。",
            "speaker": "B",
            "start_sec": 12.48,
            "end_sec": 14.12,
            "emotion": ["Sad"],
        },
        {
            "utt_id": "anjia_1_5",
            "text": "你发了两天传单了，有没有意向客户啊？要到人家电话没有？",
            "speaker": "A",
            "start_sec": 14.12,
            "end_sec": 21.92,
            "emotion": ["Anger"],
        },
    ],
    "key_frames": [
        "anjia_1_1_frame_18.jpg",
        "anjia_1_1_frame_35.jpg",
        "anjia_1_1_frame_52.jpg",
        "anjia_1_2_frame_100.jpg",
        "anjia_1_2_frame_131.jpg",
        "anjia_1_2_frame_163.jpg",
        "anjia_1_3_frame_224.jpg",
        "anjia_1_3_frame_253.jpg",
        "anjia_1_3_frame_282.jpg",
        "anjia_1_4_frame_322.jpg",
        "anjia_1_4_frame_332.jpg",
        "anjia_1_4_frame_342.jpg",
        "anjia_1_5_frame_401.jpg",
        "anjia_1_5_frame_450.jpg",
        "anjia_1_5_frame_499.jpg",
    ],
    "emo_change": {
        "start_idx": 0,
        "end_idx": 4,
        "from_emotion": ["Neutral"],
        "to_emotion": ["Anger"],
    },
    "rationale": {
        "stimulus": {"textual": "B表示自己不懂装修", "visual": None},
        "appraisal": "A认为B工作态度不好，不但没有为公司带来意向客户，还不想去监工装修",
        "response": "A感到愤怒",
    },
}
similary_samples = [
    {
        "dialogue": "A [Neutral]: 你先把衣服换了，然后去这个地址。\nA [Neutral]: 一会儿会有装修公司的人过去，你去监工，看看有什么可以帮忙的。\nB [Sad]: 这个活也派我干呀？\nB [Sad]: 装修我不懂。\nA [Anger]: 你发了两天传单了，有没有意向客户啊？要到人家电话没有？",
        "rationale": {
            "stimulus": {"textual": "B表示自己不懂装修", "visual": None},
            "appraisal": "A认为B工作态度不好，不但没有为公司带来意向客户，还不想去监工装修",
            "response": "A感到愤怒",
        },
    },
    {
        "dialogue": "A [Neutral]: 你先把衣服换了，然后去这个地址。\nA [Neutral]: 一会儿会有装修公司的人过去，你去监工，看看有什么可以帮忙的。\nB [Sad]: 这个活也派我干呀？\nB [Sad]: 装修我不懂。\nA [Anger]: 你发了两天传单了，有没有意向客户啊？要到人家电话没有？\nB [Sad]: 你不就光让我发传单吗？没让我要电话呀!",
        "rationale": {
            "stimulus": {
                "textual": "A对B没能积累意向客户表示不满",
                "visual": "A双手环抱胸前瞪着B",
            },
            "appraisal": "B认为A只让自己去发传单，没有让自己去获取客户电话",
            "response": "B感到委屈并且有点生气",
        },
    },
    {
        "dialogue": "A [Neutral]: 其实，我还真舍不得让你去伺候媳妇。\nA [Neutral]: 有你在店里，能帮我分担不少呢，我这负担也小多了。\nA [Neutral]: 额，帮我搭把手、干活、说个话，这...\nA [Happy]: 男女搭配，干活不累嘛，呵呵。\nB [Sad]: 亲家伺候媳妇，那倒对。\nB [Sad]: 我也不是她亲妈，不知道她爱吃什么，不爱吃什么。\nB [Sad]: 要是伺候不好，她不痛快，我也不痛快。\nA [Happy]: 你这么想就对了，呵呵。\nA [Happy]: 咱们做父母的，不就是为儿女做贡献的吗，是不是？\nB [Sad]: 年底咱回去吧，回老家。\nB [Sad]: 这夏天还能凑合，冬天你那腰根本就不行。\nA [Surprise]: 我，我回老家？",
        "rationale": {
            "stimulus": {"textual": "B表示要和A年底回老家", "visual": None},
            "appraisal": "A没想到B会突然提出要回老家，这让他感到意外",
            "response": "A有点惊讶",
        },
    },
]
# 生成最终结果
final_result = sample_prompt.build_prompt(sample, similary_samples)
print(final_result)
