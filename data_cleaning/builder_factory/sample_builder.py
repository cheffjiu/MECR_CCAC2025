from typing import Dict, List, Any
from parsers.utterance_parser import UtteranceParser
from parsers.change_parser import ChangeIntervalParser
from strategies.frame_sampler import FrameSamplerStrategy
from utils.video_processor import VideoProcessor
from utils.time_utils import TimeUtils


class SampleBuilder:
    """样本构建器（链式调用模式）"""

    def __init__(
        self,
        time_utils: TimeUtils = TimeUtils(),
        video_processor: VideoProcessor = VideoProcessor(),
    ):
        self.time_utils = time_utils
        self.video_processor = video_processor
        self.reset()

        # 依赖注入解析器
        self.utterance_parser = UtteranceParser(time_utils)
        self.change_parser = ChangeIntervalParser()

    def reset(self) -> None:
        """重置构建状态"""
        self.sample: Dict[str, Any] = {
            "key_frames": {"relevant_frames": []},
            "change_intervals": [],
        }

    def with_basic_info(self, sample_id: str, video_path: str) -> "SampleBuilder":
        """设置基本信息（修正视频路径）"""
        # 拼接正确的视频路径（添加memoconv_convs目录）
        correct_video_path = f"data/MECR_CCAC2025/memoconv_convs/{video_path}"
        self.sample.update({
            "id": sample_id, 
            "video_path": correct_video_path,  # 使用修正后的路径
            "rationale": {}
        })
        return self

    def with_utterances(self, utts_data: Dict[str, Dict]) -> "SampleBuilder":
        """解析并设置对话数据"""
        self.sample["utterances"] = self.utterance_parser.parse(utts_data)
        return self

    def with_key_frames(
        self, target_utt_ids: List[List[str]], sampler: FrameSamplerStrategy
    ) -> "SampleBuilder":
        """生成关键帧（添加ID格式转换逻辑）"""
        # 步骤1：从video_path提取视频编号（如anjia/anjia_11.mp4 → 11）
        video_filename = self.sample["video_path"].split("/")[-1]  # "anjia_11.mp4"
        video_number = video_filename.split("_")[-1].split(".")[0]  # "11"

        # 步骤2：转换target_utt_ids中的ID格式（anjia_sample35_1 → anjia_11_1）
        converted_utt_ids = []
        for pair in target_utt_ids:
            converted_pair = []
            for utt_id in pair:
                # 提取原始ID的后缀数字（如anjia_sample35_1 → "1"）
                suffix_num = utt_id.split("_")[-1]
                # 生成实际utts中的ID（anjia_11_1）
                converted_id = f"anjia_{video_number}_{suffix_num}"
                converted_pair.append(converted_id)
            converted_utt_ids.append(converted_pair)

        # 步骤3：使用转换后的ID调用采样器
        frames = sampler.sample(
            self.sample["utterances"],
            converted_utt_ids,  # 传递转换后的ID
            self.sample["video_path"],
            self.video_processor,
        )
        self.sample["key_frames"]["relevant_frames"] = frames
        return self

    def with_change_intervals(
        self, target_utt_ids: List[List[str]], target_emos: List[List[str]]
    ) -> "SampleBuilder":
        """解析情感变化区间（添加ID格式转换逻辑）"""
        # 步骤1：从video_path提取视频编号（如anjia/anjia_1.mp4 → 1）
        video_filename = self.sample["video_path"].split("/")[-1]  # "anjia_1.mp4"
        video_number = video_filename.split("_")[-1].split(".")[0]  # "1"

        # 步骤2：转换target_utt_ids中的ID格式（anjia_sample1_1 → anjia_1_1）
        converted_utt_ids = []
        for pair in target_utt_ids:
            converted_pair = []
            for utt_id in pair:
                # 提取原始ID的后缀数字（如anjia_sample1_1 → "1"）
                suffix_num = utt_id.split("_")[-1]
                # 生成实际utts中的ID（anjia_1_1）
                converted_id = f"anjia_{video_number}_{suffix_num}"
                converted_pair.append(converted_id)
            converted_utt_ids.append(converted_pair)

        # 步骤3：使用转换后的ID调用解析器
        self.sample["change_intervals"] = self.change_parser.parse(
            self.sample["utterances"],
            converted_utt_ids,  # 传递转换后的ID
            target_emos
        )
        return self

    def with_rationale(self, rationale_data: Dict[str, Any]) -> "SampleBuilder":
        """设置推理数据"""
        self.sample["rationale"] = rationale_data
        return self

    def build(self) -> Dict[str, Any]:
        """返回完整样本（按指定字段名和顺序排列）"""
        # 处理utterances中的id重命名（id → utt_id）
        renamed_utterances = []
        for utt in self.sample["utterances"]:
            renamed_utt = {
                "utt_id": utt["id"],
                "text": utt["text"],
                "speaker": utt["speaker"],
                "start_sec": utt["start_sec"],
                "end_sec": utt["end_sec"],
                "emotion": utt["emotion"]
            }
            renamed_utterances.append(renamed_utt)

        # 合并情感变化区间（从第一个区间的start到最后一个区间的end）
        if self.sample["change_intervals"]:
            first_interval = self.sample["change_intervals"][0]
            last_interval = self.sample["change_intervals"][-1]
            merged_emo_change = {
                "start_idx": first_interval["start_idx"],
                "end_idx": last_interval["end_idx"],
                "from_emotion": first_interval["from_emotion"],
                "to_emotion": last_interval["from_emotion"]  # 根据用户需求，取最后区间的from_emotion作为to_emotion
            }
        else:
            merged_emo_change = {}  # 无变化区间时返回空对象

        return {
            "sample_id": self.sample["id"],
            "video_path": self.sample["video_path"],
            "utterances": renamed_utterances,
            "key_frames": self.sample["key_frames"]["relevant_frames"],
            "emo_change": merged_emo_change,  # 改为合并后的单个对象
            "rationale": self.sample["rationale"]
        }
