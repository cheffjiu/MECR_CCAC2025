import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from fusion_feature_dataset import FusionFeatureDataset
from fusion_model import CrossModalAttention
from build_emotion_graph import build_emotion_graph
from GAT_modal import EmotionGraphEncoder
from collate_fn import collate_to_graph_batch

# 配置路径（根据实际路径修改）
config = {
    "feature_root": "/Users/cjh/Desktop/file/AI/MECR_CCAC2025/data/feature/demo",
    "json_path": "demo_cleaned.json",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 修改后的数据集类
class RealDataFusionDataset(FusionFeatureDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # 添加对齐验证
        sid = sample['sample_id']
        mapping_path = os.path.join(self.feature_root, sid, "text_feature/id_mapping.json")
        with open(mapping_path) as f:
            mapping = json.load(f)
            
        assert len(mapping) == len(sample['utterances']), \
            f"特征与话语数量不匹配 {len(mapping)} vs {len(sample['utterances'])}"
        
        return sample

# 初始化适配真实维度的模型
class RealDataCrossModalAttention(CrossModalAttention):
    def __init__(self):
        super().__init__(
            d_t=768,  # BERT特征维度
            d_v=512,  # 视频特征有效维度
            d_model=512,  # 融合维度
            num_heads=8,
            num_layers=3
        )

# 测试流程
def real_data_test():
    # 1. 加载数据集
    dataset = RealDataFusionDataset(
        feature_root=config["feature_root"],
        json_path=config["json_path"],
        mode="train"
    )
    
    # 2. 实例化适配模型
    fusion_model = RealDataCrossModalAttention().to(config["device"])
    gat_model = EmotionGraphEncoder(in_dim=512).to(config["device"])  # 输入维度匹配融合输出
    
    # 3. 数据加载器
    def custom_collate(batch):
        return collate_to_graph_batch(
            batch_list=batch,
            fusion_model=fusion_model,
            graph_builder=build_emotion_graph,
            tokenizer=None,
            with_prompt=False
        )
    
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)
    
    # 4. 执行测试
    for batch in dataloader:
        # 设备转移
        batch = batch.to(config["device"])
        
        # 验证特征维度
        sample = batch[0]  # 取第一个样本
        print(f"原始特征验证：")
        print(f"文本特征形状: {sample.t_feats.shape} (应为 [N,768])")
        print(f"视频特征形状: {sample.v_feats.shape} (应为 [N,512])")
        
        # 图神经网络前向传播
        h_change, all_nodes = gat_model(batch)
        
        # 结果验证
        print("\n融合及图网络验证：")
        print(f"融合输出维度: {batch.x.shape} (应为 [总节点数, 512])")
        print(f"超级节点索引: {batch.super_idx}")
        print(f"h_change形状: {h_change.shape} (应为 [batch_size, 4096])")
        print(f"图节点总数: {batch.num_nodes}")
        
        break  # 只测试第一个batch

if __name__ == "__main__":
    # 样本特征验证
    sample_path = os.path.join(config["feature_root"], "anjia_sample1/text_feature/anjia_sample1.pt")
    text_feat = torch.load(sample_path)
    print(f"文本特征实际维度: {text_feat.shape} (应为 [5,768])")
    
    video_path = os.path.join(config["feature_root"], "anjia_sample1/video_feature/anjia_sample1_video.pt")
    video_feat = torch.load(video_path)
    print(f"视频特征实际维度: {video_feat.shape} (应为 [5,513])")
    
    # 执行完整测试
    real_data_test()