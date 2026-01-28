import cv2 as cv
from ultralytics import YOLO
import supervision as sv
import time  # 添加这个导入

from utils.downloader import download_video_if_needed
from utils.constants import IMAGE_POINTS, WORLD_POINTS
from config.settings import *

from modules.mapping import Cam2WorldMapper
from modules.speedometer import Speedometer
from modules.annotators import get_annotators
from zone.zone_trigger import create_zone

def main():
    source_video = download_video_if_needed()
    video_info = sv.VideoInfo.from_video_path(source_video)
    FPS = video_info.fps
    WIDTH = round(video_info.width / 32) * 32
    HEIGHT = round(video_info.height / 32) * 32
    
    # 计算每帧应该等待的时间（毫秒）
    frame_delay = int(1000 / FPS) if FPS > 0 else 33  # 默认30fps

    cap = cv.VideoCapture(source_video)

    # Setup
    mapper = Cam2WorldMapper()
    mapper.find_perspective_transform(IMAGE_POINTS, WORLD_POINTS)
    zone = create_zone()
    annotators = get_annotators(FPS)
    speedometer = Speedometer(mapper, FPS, MPS_TO_KPH)
    model = YOLO(MODEL_PATH)

    print("="*60)
    print("车辆速度检测系统已启动")
    print("="*60)
    print("退出方式:")
    print("  - 在视频窗口按 'q' 键退出")
    print("  - 在视频窗口按 'ESC' 键退出")
    print("  - 关闭视频窗口退出")
    print("  - 视频播放完毕自动退出")
    print("="*60)

    while True:
        start_time = time.time()  # 记录开始时间
        
        ret, frame = cap.read()
        if not ret:
            break

        result = model.track(
            frame,
            classes=CLASSES_TO_TRACK,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=(HEIGHT, WIDTH),
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )

        detections = sv.Detections.from_ultralytics(result[0])
        detections = detections[zone.trigger(detections=detections)]
        trace_ids = detections.tracker_id
        labels = []

        for trace_id in trace_ids:
            trace = annotators["trace"].trace.get(trace_id)
            speedometer.update_with_trace(trace_id, trace)
            speed = speedometer.get_current_speed(trace_id)
            labels.append(f"#Vehicle Id:{trace_id} Speed:{speed} km/h")

        # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # frame = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR)

        frame = annotators["bbox"].annotate(frame, detections)
        frame = annotators["trace"].annotate(frame, detections)
        frame = annotators["label"].annotate(frame, detections, labels=labels)

        cv.imshow("Vehicle Speed Estimation - YOLOv11", frame)
        
        # 计算处理时间，动态调整等待时间
        processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
        wait_time = max(1, int(frame_delay - processing_time))
        
        # 支持多种退出方式
        key = cv.waitKey(wait_time) & 0xFF
        if key == ord("q") or key == 27:  # 'q' 键或 ESC 键
            print("\n用户主动退出程序")
            break
        
        # 检查窗口是否被关闭
        if cv.getWindowProperty("Vehicle Speed Estimation - YOLOv11", cv.WND_PROP_VISIBLE) < 1:
            print("\n视频窗口已关闭，退出程序")
            break

    cap.release()
    cv.destroyAllWindows()
    print("程序已正常退出")

if __name__ == "__main__":
    main()