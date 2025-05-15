import cv2
from typing import Optional


class VideoProcessor:
    """视频处理类（使用上下文管理器管理资源）"""

    def __init__(self):
        self.cap = cv2.VideoCapture()  # 初始化视频捕捉对象
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap.isOpened():
            self.cap.release()  # 确保释放资源
    
    def get_fps(self, video_path):
        self.cap.open(video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps
