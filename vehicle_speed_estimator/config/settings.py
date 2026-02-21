#default for test
#win
# VIDEO_PATH = "assets\test3.mp4"

#mac
VIDEO_PATH = "assets/crash2.mp4"
# VeronaRd

#no downloaded video
VIDEO_URL = "https://drive.google.com/uc?export=download&id=1fYb05GW0sWeI1EiTfcbjOtGBrdIuPq4U"
VIDEO_NAME = "test.mp4"
VIDEO_DOWNLOAD_PATH = "assets"  # 下载视频保存的文件夹

# 模型配置
MODEL_PATH = "models/yolo11s.pt"
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

# 监测区域可视化开关（边界线 + 文本标签 + A/B/C/D）
SHOW_MONITORING_AREA = True
MONITORING_AREA_LINE_COLOR = (170, 230, 255)  # BGR 浅黄
MONITORING_AREA_LINE_THICKNESS = 3
MONITORING_AREA_LABEL_TEXT = "Monitoring Area"
MONITORING_AREA_LABEL_FONT_SCALE = 0.9
MONITORING_AREA_LABEL_THICKNESS = 3
# 文本位置直观调参（推荐每次改 10~20 像素）
# OFFSET_X > 0 向右，OFFSET_X < 0 向左；常用: +20 / +40 / +60
# OFFSET_Y > 0 向下，OFFSET_Y < 0 向上
MONITORING_AREA_LABEL_OFFSET_X = 400
MONITORING_AREA_LABEL_OFFSET_Y = 24  # 文本基线位于 ROI 下方的偏移
MONITORING_AREA_LABEL_MARGIN_X = 8
MONITORING_AREA_LABEL_MARGIN_Y = 8

# 车道编号配置
# LANE_BOUNDARIES_X 为图像坐标系下的车道分界线 x 值，需按从左到右递增。
# 例如 [900, 1300] 表示共 3 根车道：(<900)=1, [900,1300)=2, (>=1300)=3。
ENABLE_LANE_ASSIGNMENT = True
LANE_ASSIGNMENT_MODE = "line_segments"

# 车道边界线（按从左到右顺序），每条边界线由两个点定义: [(x1, y1), (x2, y2)]
# 使用 line_segments 模式时优先使用该配置；为空时自动回退到 LANE_BOUNDARIES_X。
LANE_BOUNDARY_LINES = [[(583, 261), (1110, 1220)], [(646, 259), (1374, 1189)], [(716, 258), (1626, 1155)], [(772, 259), (1823, 1132)], [(820, 245), (2061, 1108)], [(1072, 247), (2367, 825)], [(1133, 244), (2460, 785)], [(1170, 233), (2546, 739)]]
SHOW_LANE_BOUNDARY_LINES = False
LANE_LINE_COLOR = (144, 238, 144)  # BGR 浅绿色
LANE_LINE_THICKNESS = 2

# 应急车道配置（按边界线索引定义区间，索引从 1 开始）
# 新配置优先：EMERGENCY_LANE_RULES
# - {"type": "left_of", "boundary": 1}    => 第1条边界线左侧（第1车道）
# - {"type": "between", "left": 1, "right": 2} => 第1和第2条边界线之间
# - {"type": "right_of", "boundary": 4}   => 第4条边界线右侧
# 当 EMERGENCY_LANE_RULES 为空时，回退使用 legacy 字段 EMERGENCY_LANE_BETWEEN_BOUNDARIES。
EMERGENCY_LANE_ENABLED = True
EMERGENCY_LANE_RULES = [
    {"type": "left_of", "boundary": 1},
    {"type": "right_of", "boundary": 8}
]
# legacy 回退字段：例如 (1, 2) 表示第1和第2条边界线之间的车道是应急车道。
EMERGENCY_LANE_BETWEEN_BOUNDARIES = (1, 2)

# 应急车辆可视化配置
HIGHLIGHT_EMERGENCY_VEHICLES = True
EMERGENCY_BBOX_COLOR = (0, 0, 255)  # BGR 红色
EMERGENCY_BBOX_THICKNESS = 2

# 超速检测与可视化配置
SPEED_LIMIT_MPH = 30
MAX_VALID_SPEED_MPH = 95
HIGHLIGHT_SPEEDING_VEHICLES = True
SPEEDING_BBOX_COLOR = (0, 0, 255)  # BGR 红色
SPEEDING_BBOX_THICKNESS = 2


