import json
import torch
import os
from transformers import BertTokenizer, BertModel
from accelerate import Accelerator
from tqdm import tqdm
import logging

# 初始化日志和加速器
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
accelerator = Accelerator()
device = accelerator.device

# 设备优化配置
device_type = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch_device = torch.device(device_type)
logging.info(f"Active device: {device_type.upper()}")

# 加载模型和分词器（自动处理设备）
with accelerator.main_process_first():
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese").to(torch_device)

# 模型配置优化
model = accelerator.prepare(model)
model.eval()
for param in model.parameters():
    param.requires_grad = False

# 路径配置
current_file_path = os.path.abspath(__file__)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), ".."))
json_path = os.path.join(

    workspace_root, "data/processed/demo_cleaned/demo_cleaned.json"
)

# 处理数据
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 内存优化上下文
with torch.inference_mode(), accelerator.autocast(), torch.backends.cuda.sdp_kernel():
    for sample in tqdm(data, desc="提取文本特征", total=len(data)):
        sample_id = sample["sample_id"]
        utterances = sample["utterances"]

        # 生成合并文本（使用生成器表达式减少内存占用）
        merged_texts = [
            f"utt_id: {utt['utt_id']}, speaker: {utt['speaker']}, emotion: {' '.join(utt['emotion'])}, text: {utt['text']}"
            for utt in utterances
        ]

        # 特征提取流水线
        inputs = tokenizer(
            merged_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}  # 显式数据移动

        outputs = model(**inputs)
        t_feats = outputs.pooler_output

        # 特征保存（添加异步操作）
        feature_dir = os.path.join(
            workspace_root, "data/feature/demo", sample_id, "text_feature"
        )
        os.makedirs(feature_dir, exist_ok=True)

        # 使用异步保存和单精度存储
        with accelerator.autocast():
            torch.save(t_feats.float(), os.path.join(feature_dir, f"{sample_id}.pt"))

        # 内存清理
        if device_type == "cuda":
            torch.cuda.empty_cache()

logging.info("文本特征提取完成，所有结果已保存")
