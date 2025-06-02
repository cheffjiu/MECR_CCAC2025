from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import os
import torch


class FeatureLoader(ABC):
    @abstractmethod
    def load_features(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        pass


# pytroch tensor dataset
class PtFeatureLoader(FeatureLoader):
    def __init__(self, feature_root: str):
        super().__init__()
        self.feature_root = feature_root

    def load_features(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # 获取样本id
        sid = sample["sample_id"]
        # 构建特征路径
        t_feats_path = os.path.join(self.feature_root, sid, "text_feature", f"{sid}.pt")
        v_feats_path = os.path.join(
            self.feature_root, sid, "video_feature", f"{sid}_video.pt"
        )
        # 加载特征到CPU，由Accelerate负责设备转移
        t_feats = torch.load(t_feats_path, map_location="cpu")
        v_feats = torch.load(v_feats_path, map_location="cpu")
        # 返回特征
        return (t_feats, v_feats)
