from typing import Dict, Any
from .sample_builder import SampleBuilder
from strategies.frame_sampler import FrameSamplerStrategy


class SampleFactory:
    """样本工厂（策略模式注入）"""

    def __init__(self, default_sampler: FrameSamplerStrategy):
        self.default_sampler = default_sampler

    def create_sample(self, sample_id: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """根据原始数据创建标准化样本"""
        return (
            SampleBuilder()
            .with_basic_info(sample_id, raw_data["video_path"])
            .with_utterances(raw_data["utts"])
            .with_key_frames(raw_data["target_change_utt_ids"], self.default_sampler)
            .with_change_intervals(
                raw_data["target_change_utt_ids"], raw_data["target_change_emos"]
            )
            .with_rationale(raw_data["rationale"])
            .build()
        )
