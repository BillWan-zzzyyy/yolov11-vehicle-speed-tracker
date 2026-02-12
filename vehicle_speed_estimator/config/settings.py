#default for test
#win
# VIDEO_PATH = "assets\test3.mp4"

#mac
VIDEO_PATH = "assets/VeronaRd.mp4"

#no downloaded video
VIDEO_URL = "https://drive.google.com/uc?export=download&id=1fYb05GW0sWeI1EiTfcbjOtGBrdIuPq4U"
VIDEO_NAME = "test.mp4"
VIDEO_DOWNLOAD_PATH = "assets"  # 下载视频保存的文件夹

# 模型配置
MODEL_PATH = "models/yolo11n.pt"
MPS_TO_MPH = 2.2369362921
CLASSES_TO_TRACK = [0, 1, 2, 5, 7]  #person, bicycle， car, bus, truck
CONFIDENCE_THRESHOLD = 0.25

# 实时录制配置
RECORD_OUTPUT_VIDEO = False
RECORD_OUTPUT_DIR = "results"
RECORD_OUTPUT_PREFIX = "tracked_video"
RECORD_OUTPUT_CODEC = "mp4v"
RECORD_OUTPUT_QUEUE_SIZE = 100
RECORD_OUTPUT_DROP_FRAMES = True

# 车辆数据保存开关（CSV/JSON/摘要）
SAVE_VEHICLE_DATA = False

# 监测区域可视化开关（浅灰阴影 + A/B/C/D）
SHOW_MONITORING_AREA = True
