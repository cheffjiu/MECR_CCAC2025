import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator
from tqdm import tqdm
import logging

# 初始化加速器和日志
accelerator = Accelerator()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载模型处理器和配置设备
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

# 模型优化配置
model = accelerator.prepare(model.eval())
for param in model.parameters():
    param.requires_grad = False

# 路径配置
current_file_path = os.path.abspath(__file__)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), ".."))
json_path = os.path.join(workspace_root, "data/processed/demo_cleaned/demo_cleaned.json")

# 内存优化上下文
with accelerator.main_process_first(), torch.inference_mode(), accelerator.autocast():
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for sample in  tqdm(data,desc="提取视觉特征",total=len(data)):
        sample_id = sample['sample_id']
        utterances = sample['utterances']
        key_frames = sample['key_frames']

        # 初始化张量（自动设备分配）
        num_utterances = len(utterances)
        v_feats = torch.zeros((num_utterances, 512), device=accelerator.device)
        vis_mask = torch.zeros(num_utterances, dtype=torch.uint8, device=accelerator.device)

        for key_frame in key_frames:
            utt_id = key_frame.split('_frame_')[0]
            utt_index = next((i for i, utt in enumerate(utterances) if utt['utt_id'] == utt_id), None)
            
            if utt_index is not None:
                try:
                    # 图像处理流水线
                    image_path = os.path.join(workspace_root, 'data/processed/demo_cleaned/keyframes', sample_id, key_frame)
                    image = Image.open(image_path)
                    
                    # 使用加速器处理输入
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}  # 显式数据移动
                    
                    # 特征提取
                    # 在模型准备后添加设备类型定义
                    device_type = accelerator.device.type  # 新增这行获取设备类型
                    
                    # 修改混合精度上下文调用
                    with torch.amp.autocast(
                        device_type=device_type,  # 使用已定义的变量
                        dtype=torch.float16,
                        enabled=accelerator.mixed_precision == 'fp16'
                    ):
                        image_features = model.get_image_features(**inputs)
                        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    
                    # 特征更新逻辑
                    current_feat = accelerator.gather(image_features).squeeze()
                    if vis_mask[utt_index] == 1:
                        v_feats[utt_index] = (v_feats[utt_index] + current_feat) / 2
                    else:
                        v_feats[utt_index] = current_feat
                        vis_mask[utt_index] = 1
                        
                except Exception as e:
                    logging.error(f"处理 {image_path} 失败: {str(e)}")
                    continue

        # 保存优化
        video_features = torch.cat([vis_mask.unsqueeze(1), v_feats], dim=1)
        video_feature_dir = os.path.join(workspace_root, "data/feature/demo", sample_id, "video_feature")
        os.makedirs(video_feature_dir, exist_ok=True)
        
        # 使用单精度存储
        torch.save(video_features.float(), os.path.join(video_feature_dir, f"{sample_id}_video.pt"))

logging.info("视觉特征提取完成")
    