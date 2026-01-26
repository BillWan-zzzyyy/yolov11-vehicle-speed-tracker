import gdown
import os
from config.settings import VIDEO_PATH, VIDEO_URL, VIDEO_NAME, VIDEO_DOWNLOAD_PATH

def download_video_if_needed():
    """
    获取视频路径
    如果 VIDEO_PATH 已设置，直接返回本地路径
    否则从网络下载视频
    """
    # 如果配置了本地视频路径，直接使用
    if VIDEO_PATH:
        # 检查文件是否存在（支持相对路径和绝对路径）
        if os.path.exists(VIDEO_PATH):
            return VIDEO_PATH
        else:
            raise FileNotFoundError(f"视频文件不存在: {VIDEO_PATH}")
    
    # 否则从网络下载
    path = VIDEO_DOWNLOAD_PATH
    os.makedirs(path, exist_ok=True)
    local_path = os.path.join(path, VIDEO_NAME)
    return gdown.cached_download(VIDEO_URL, local_path)
