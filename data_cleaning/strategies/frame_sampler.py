from abc import ABC, abstractmethod
from typing import List, Dict, Any
from utils.video_processor import VideoProcessor
import os
import cv2
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FrameSamplerStrategy(ABC):
    """帧采样策略抽象类"""

    @abstractmethod
    def sample(
        self,
        utterances: List[Dict],
        target_utt_ids: List[List[str]],
        video_path: str,
        video_processor: VideoProcessor,
    ) -> List[str]:
        pass


class ContextWindowSampler(FrameSamplerStrategy):
    """上下文窗口采样策略（实现N=3，每utterance采样2帧）"""

    def __init__(self, window_size: int = 3, frames_per_utt: int = 2):
        self.window_size = window_size
        self.frames_per_utt = frames_per_utt

    def sample(self, utterances: List[Dict], target_utt_ids: List[List[str]], video_path: str, video_processor: VideoProcessor) -> List[str]:
        # 获取最后一个有效情感变化对（修正：取变化对的结束utterance作为U_change）
        if not target_utt_ids:
            return []
        
        last_change_pair = target_utt_ids[-1]
        if len(last_change_pair) < 2:  # 确保变化对包含start和end两个id
            return []
            
        u_change_id = last_change_pair[1]  # 取变化对中的结束utterance（情感变化发生的utterance）
        
        # 定位目标utterance索引（添加日志辅助调试）
        u_change_idx = next((i for i, u in enumerate(utterances) if u["id"] == u_change_id), -1)
        if u_change_idx == -1:
            logging.warning(f"警告：未找到utterance {u_change_id} 在utterances列表中的索引")  # 添加日志提示
            return []
            
        # 计算上下文窗口范围（N=3）
        start_idx = max(0, u_change_idx - 2)  # 包含当前和前2个utterance
        end_idx = u_change_idx + 1
        
        # 实际采样逻辑（修正视频路径拼接）
        frames = []
        #获取当前项目的根目录
        current_file_path = os.path.abspath(__file__)
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path),"..",".."))
        # logging.info("当前项目的根目录为：{}".format(workspace_root))
        with VideoProcessor() as vp:
            #获取视频文件path
            vedio_data_path=os.path.join(workspace_root, "data/MECR_CCAC2025/memoconv_convs")
            # logging.info(f"视频文件路径为：{vedio_data_path}")
            full_video_path = os.path.join(vedio_data_path,  video_path)
            # logging.info(f"完整视频路径为：{full_video_path}")
            if not vp.cap.open(full_video_path):
                logging.error(f"无法打开视频文件: {full_video_path}")
                return []
            
            for utt in utterances[start_idx:end_idx]:
                start_sec = utt["start_sec"]
                end_sec = utt["end_sec"]
                duration = end_sec - start_sec
                
                # 处理持续时间为0的情况
                if duration <= 0:
                    utt_id = utt["id"]
                    start_frame = self._convert_sec_to_frame(start_sec)
                    logging.warning(f"强制采集零持续时间帧: {utt_id}_frame_{start_frame}")
                    frames.append(f"{utt_id}_frame_{start_frame}.jpg")
                    continue
                
                # 添加视频帧数边界检查
                total_frames = int(vp.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 每utterance均匀采样2帧（修正时间间隔计算）
                # 采样逻辑（添加越界修正）
                sample_times = [start_sec + duration * (i+1)/(self.frames_per_utt + 1) for i in range(self.frames_per_utt)]
                valid_frames = []
                
                for t in sample_times:
                    frame_num = self._convert_sec_to_frame(t)
                    # 越界修正逻辑
                    if frame_num >= total_frames:
                        adjusted_frame = total_frames - 1  # 最后一帧
                        logging.warning(f"调整越界帧 {frame_num} → {adjusted_frame}")
                        valid_frames.append(adjusted_frame)
                    else:
                        valid_frames.append(frame_num)
                
                # 补足缺失帧
                while len(valid_frames) < self.frames_per_utt:
                    last_valid = valid_frames[-1] if valid_frames else (total_frames - 1)
                    new_frame = max(0, last_valid - 1)  # 向前取帧
                    valid_frames.append(new_frame)
                    logging.warning(f"补足缺失帧：{new_frame}")

                for frame_num in valid_frames:
                    vp.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, _ = vp.cap.read()
                    if ret:
                        frames.append(f"{utt['id']}_frame_{frame_num}.jpg")
                    else:
                        logging.warning(f"警告：无法读取帧{utt['id']}_frame_{frame_pos}.jpg {frame_pos}")
        
        return frames

    def _convert_sec_to_frame(self, seconds: float) -> int:
        """秒转帧号（基于25fps）"""
        return int(seconds * 25)
