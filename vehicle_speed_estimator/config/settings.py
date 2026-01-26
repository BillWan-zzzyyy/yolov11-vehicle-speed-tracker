# 视频路径配置
# 方式1: 使用本地视频文件（推荐）- 直接指定本地视频文件的完整路径
VIDEO_PATH = r"D:\OneDrive - UW-Madison\Project\alpha_stage_2\yolov11-vehicle-speed-tracker\vehicle_speed_estimator\assets\testcrash.mp4"

# 方式2: 从网络下载视频 - 如果 VIDEO_PATH 为 None，则使用以下配置下载
VIDEO_URL = "https://drive.google.com/uc?export=download&id=1fYb05GW0sWeI1EiTfcbjOtGBrdIuPq4U"
VIDEO_NAME = "test.mp4"
VIDEO_DOWNLOAD_PATH = "assets"  # 下载视频保存的文件夹

# 模型配置
MODEL_PATH = "models/yolo11n.pt"
MPS_TO_KPH = 3.6
CLASSES_TO_TRACK = [2, 5, 7]  # car, bus, truck
CONFIDENCE_THRESHOLD = 0.4
