# main.py
import json
from pipeline.data_pipeline import DataCleaningPipeline


def main():
    # 初始化清洗管道
    pipeline = DataCleaningPipeline()

    # 处理训练数据（添加项目根目录前缀）
    train_samples = pipeline.process_file(
        "data/MECR_CCAC2025/demo.json"
    
    )

    # 输出示例结果（仅显示第一个样本）
    if train_samples:
        print(json.dumps(train_samples[0], indent=2, ensure_ascii=False))
        print(f"Processed {len(train_samples)} samples successfully")


if __name__ == "__main__":
    main()
