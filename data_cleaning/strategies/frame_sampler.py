from abc import ABC, abstractmethod
from typing import List, Dict, Any
from utils.video_processor import VideoProcessor
import os
import cv2


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
            print(f"警告：未找到情感变化utterance ID: {u_change_id}")  # 添加调试日志
            return []
            
        # 计算上下文窗口范围（N=3）
        start_idx = max(0, u_change_idx - 2)  # 包含当前和前2个utterance
        end_idx = u_change_idx + 1
        
        # 实际采样逻辑（修正视频路径拼接）
        frames = []
        with VideoProcessor() as vp:
            # 使用系统提供的workspace_folder拼接正确路径（避免重复目录）
            workspace_folder = "/Users/cjh/Desktop/file/AI/MECR_CCAC2025"
            full_video_path = os.path.join(workspace_folder,  video_path)
            if not vp.cap.open(full_video_path):
                print(f"错误：无法打开视频文件 {full_video_path}")  # 添加错误日志
                return []
            
            for utt in utterances[start_idx:end_idx]:
                duration = utt["end_sec"] - utt["start_sec"]
                if duration <= 0:
                    print(f"警告：utterance {utt['id']} 持续时间无效（{utt['start_sec']}-{utt['end_sec']}）")
                    continue
                    
                # 每utterance均匀采样2帧（修正时间间隔计算）
                sample_times = [utt["start_sec"] + duration * (i+1)/(self.frames_per_utt + 1) for i in range(self.frames_per_utt)]
                
                for t in sample_times:
                    frame_pos = int(t * 25)  # 25fps转换
                    vp.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, _ = vp.cap.read()
                    if ret:
                        frames.append(f"{utt['id']}_frame_{frame_pos}.jpg")
                    else:
                        print(f"警告：utterance {utt['id']} 帧 {frame_pos} 读取失败")  # 添加帧读取失败日志
        
        return frames

    def _extract_frames(self, video_path: str, duration: float, fps: float) -> list:
        """实际帧提取逻辑"""
        frames = []
        total_frames = int(duration * fps)
        
        with VideoProcessor() as vp:
            vp.cap.open(video_path)
            for frame_idx in range(0, total_frames, self.sample_interval):
                vp.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = vp.cap.read()
                if ret:
                    frames.append({
                        "frame_index": frame_idx,
                        "timestamp": frame_idx / fps,
                        "frame_data": cv2.imencode('.jpg', frame)[1].tobytes().hex()
                    })
        return frames
