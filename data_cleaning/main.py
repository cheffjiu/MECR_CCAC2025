
import json
import os
from pipeline.data_pipeline import DataCleaningPipeline
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    #获取当前项目的根目录
    current_file_path = os.path.abspath(__file__)
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path),".."))
    #logging.info("当前项目的根目录为：{}".format(workspace_root))

    # 初始化清洗管道
    pipeline = DataCleaningPipeline()

    # 处理训练数据（添加项目根目录前缀）
    # file_path = "data/MECR_CCAC2025/train.json"#训练集json文件路径
    file_path = "data/MECR_CCAC2025/val.json"#验证集json文件路径
    # file_path = "data/MECR_CCAC2025/demo.json" #样本json文件路径
    sub_name=file_path[19:-1]
    logging.info("正在处理文件：{}".format(sub_name))
    file_path = os.path.join(workspace_root, file_path)
    train_samples = pipeline.process_file(file_path)

    # 获取文件名和创建输出目录
    file_name = os.path.basename(file_path).replace('.json', '')
    output_folder = os.path.join("data", "processed", f"{file_name}_cleaned")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{file_name}_cleaned.json")

    # 保存清洗后的 JSON 数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)

    logging.info("清洗后的数据已保存到：{}".format(output_path))
    #print(json.dumps(train_samples, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()