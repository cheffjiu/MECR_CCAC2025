from typing import List, Dict
from parsers.json_parser import JsonDataParser
from builder_factory.sample_factory import SampleFactory
from strategies.frame_sampler import ContextWindowSampler,EveryUtteranceSampler
from utils.video_processor import VideoProcessor


class DataCleaningPipeline:
    """数据清洗管道（责任链模式）"""

    def __init__(self):
        self.parser = JsonDataParser()
        self.factory = SampleFactory(EveryUtteranceSampler())  # 默认采样策略

    def process_file(self, file_path: str) -> List[Dict]:
        """处理JSON文件并返回清洗后的数据"""
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = self.parser.parse(f.read())

        cleaned_samples = []
        with VideoProcessor() as vp:  # 使用上下文管理器管理视频处理器
            for sample_id, sample_data in raw_data.items():
                # 注入视频处理器实例
                sample = self.factory.create_sample(sample_id, sample_data)
                cleaned_samples.append(sample)

        return cleaned_samples
