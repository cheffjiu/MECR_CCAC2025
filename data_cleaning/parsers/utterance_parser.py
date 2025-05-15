from typing import List, Dict
from utils.time_utils import TimeUtils


class UtteranceParser:
    """对话解析器"""

    def __init__(self, time_utils: TimeUtils = TimeUtils()):
        self.time_utils = time_utils

    def parse(self, utts_data: Dict[str, Dict]) -> List[Dict]:
        """解析对话数据并按时间排序"""
        utterances = []
        for utt_id, utt_info in utts_data.items():
            try:
                start = self.time_utils.timestamp_to_seconds(utt_info["timestamps"][0])
                end = self.time_utils.timestamp_to_seconds(utt_info["timestamps"][1])
            except (IndexError, ValueError):
                continue  # 跳过无效时间戳

            utterances.append(
                {
                    "id": utt_id,
                    "text": utt_info["utterance"],
                    "speaker": utt_info["speaker"],
                    "start_sec": start,
                    "end_sec": end,
                    "emotion": utt_info["emotion"],
                }
            )

        return sorted(utterances, key=lambda x: x["start_sec"])
