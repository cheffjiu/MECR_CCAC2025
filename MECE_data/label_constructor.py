from abc import ABC, abstractmethod
from typing import Any, Dict
import json


class LabelConstructor(ABC):
    @abstractmethod
    def build_label_from_sample(self, sample: Dict[str, Any]) -> str:
        pass


class DefaultLabelConstructor(LabelConstructor):
    def __init__(self):
        super().__init__()

    def build_label_from_sample(self, sample: Dict[str, Any]) -> str:
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



label_prompt = DefaultLabelConstructor()
data = {
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
print(label_prompt.build_label_from_sample(data))
