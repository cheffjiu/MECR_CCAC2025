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
dataset_name = "val"  
json_path = os.path.join(workspace_root, f"data/processed/{dataset_name}_cleaned/{dataset_name}_cleaned.json")

# 内存优化上下文
with accelerator.main_process_first(), torch.inference_mode(), accelerator.autocast():
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for sample in tqdm(data, desc="提取视觉特征", total=len(data)):
        sample_id = sample['sample_id']
        utterances = sample['utterances']
        key_frames = sample['key_frames']

        # 预计算utt_id到索引的映射
        utt_id_to_index = {utt['utt_id']: i for i, utt in enumerate(utterances)}

        # 初始化特征张量和帧计数器
        num_utterances = len(utterances)
        v_feats = torch.zeros((num_utterances, 512), device=accelerator.device)
        frame_counts = torch.zeros(num_utterances, dtype=torch.int32, device=accelerator.device)

        for key_frame in key_frames:
            # 从文件名解析utt_id
            utt_id = key_frame.split('_frame_')[0]
            utt_index = utt_id_to_index.get(utt_id, None)
            
            if utt_index is not None:
                try:
                    # 图像处理流水线
                    image_path = os.path.join(
                        workspace_root, 
                        f'data/processed/{dataset_name}_cleaned/keyframes', 
                        sample_id, 
                        key_frame
                    )
                    
                    # 检查文件是否存在
                    if not os.path.exists(image_path):
                        logging.warning(f"图像不存在: {image_path}")
                        continue
                    
                    image = Image.open(image_path)
                    
                    # 使用加速器处理输入
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                    
                    # 特征提取
                    device_type = accelerator.device.type
                    
                    with torch.amp.autocast(
                        device_type=device_type,
                        dtype=torch.float16,
                        enabled=accelerator.mixed_precision == 'fp16'
                    ):
                        image_features = model.get_image_features(**inputs)
                        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    
                    # 改进的特征更新逻辑 - 累积平均
                    current_feat = accelerator.gather(image_features).squeeze()
                    frame_counts[utt_index] += 1
                    count = frame_counts[utt_index].float()
                    
                    # 使用累积平均公式: new_avg = ((n-1)/n) * old_avg + (1/n) * new_value
                    v_feats[utt_index] = ((count - 1) / count) * v_feats[utt_index] + (1 / count) * current_feat
                        
                except Exception as e:
                    logging.error(f"处理 {image_path} 失败: {str(e)}")
                    continue

        # 保存特征
        video_feature_dir = os.path.join(workspace_root, f"data/feature/{dataset_name}", sample_id, "video_feature")
        os.makedirs(video_feature_dir, exist_ok=True)
        
        # 保存特征张量，形状为 [num_utterances, 512]
        #logging.info(f"保存特征张量: {v_feats.shape}")
        torch.save(v_feats.float(), os.path.join(video_feature_dir, f"{sample_id}_video.pt"))

logging.info("视觉特征提取完成")    