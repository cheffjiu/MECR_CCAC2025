from datetime import datetime
from typing import Optional, Union


class TimeUtils:
    """时间处理工具类"""

    @staticmethod
    def timestamp_to_seconds(timestamp: str) -> float:
        """将 hh:mm:ss:ff 格式的时间戳转换为秒（支持小数帧）"""
        try:
            parts = timestamp.split(":")
            h, m, s, ff = map(int, parts) if len(parts) == 4 else (*parts[:3], 0)
            total_seconds = h * 3600 + m * 60 + s + ff / 25  # 兼容原始帧率逻辑
            return round(total_seconds, 2)  # 保留两位小数
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid timestamp format: {timestamp}") from e
