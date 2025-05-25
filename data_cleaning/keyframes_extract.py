import cv2
import sys
import json
import os
import logging
from tqdm import tqdm  

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_specific_frames(video_path, frame_info_list, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"错误：无法打开视频文件 {video_path}")
        return

    for frame_info in frame_info_list:
        frame_num = frame_info['frame_num']
        file_name = frame_info['file_name']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, frame)
            #logging.info(f"已保存帧 {frame_num} 到 {output_path}") 
        else:
            logging.warning(f"警告：无法读取帧 {frame_num}")

    cap.release()

if __name__ == "__main__":
    #获取当前项目的根目录
    current_file_path = os.path.abspath(__file__)
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path),".."))
    # logging.info("当前项目的根目录为：{}".format(workspace_root))
    # 读取 JSON 文件
    json_path = os.path.join(workspace_root, "data/processed/demo_cleaned/demo_cleaned.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 关键帧输出目录
    base_output_dir = os.path.join(workspace_root, "data/processed/demo_cleaned/keyframes")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 遍历每个样本单独处理（添加主进度条）
    with tqdm(total=len(data), desc="总进度") as main_pbar:
        for item in data:
            # 动态获取每个样本的视频路径（拼接绝对路径）
            #获取视频文件路径
            video_data_path = os.path.join(workspace_root, "data/MECR_CCAC2025/memoconv_convs")
            relative_video_path = item["video_path"]
            video_path = os.path.join(video_data_path, relative_video_path)
    
            sample_id = item["sample_id"]
            frame_info_list = []
            
            # 创建样本专属输出目录
            output_dir_sample = os.path.join(base_output_dir, sample_id)
            os.makedirs(output_dir_sample, exist_ok=True)
            
            # 提取当前样本的帧信息
            for frame_name in item["key_frames"]:
                frame_num_str = frame_name.split('_')[-1].replace('.jpg', '')
                frame_info = {
                    'frame_num': int(frame_num_str),
                    # 修正文件名生成逻辑，保持原始文件名
                    'file_name': frame_name  
                }
                frame_info_list.append(frame_info)
            
            # 为当前样本提取关键帧（添加子进度条）
            with tqdm(total=len(frame_info_list), desc=f"样本 {sample_id}") as sub_pbar:
                extract_specific_frames(video_path, frame_info_list, output_dir_sample)
                sub_pbar.update(len(frame_info_list))
            
            main_pbar.update(1)