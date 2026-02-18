import cv2 as cv
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time  # 添加这个导入

from utils.downloader import download_video_if_needed
from utils.constants import IMAGE_POINTS, WORLD_POINTS
from config.settings import *

from modules.mapping import Cam2WorldMapper
from modules.speedometer import Speedometer
from modules.annotators import get_annotators
from modules.data_recorder import VehicleDataRecorder
from modules.video_recorder import AsyncVideoRecorder
from zone.zone_trigger import create_zone


def draw_monitoring_area_overlay(frame, image_points):
    """Draw only thin light-yellow boundary for ROI."""
    points = np.array(image_points, dtype=np.int32)
    if points.shape[0] != 4:
        return frame

    cv.polylines(frame, [points], True, (170, 230, 255), 1)
    
    # Add title "Monitoring Area"
    title_pos = (int(points[0][0]), int(points[0][1]) - 20)
    cv.putText(frame, "Monitoring Area", title_pos,
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (170, 230, 255), 2)

    for label, (x, y) in zip(("A", "B", "C", "D"), points):
        cv.putText(frame, label, (int(x) + 8, int(y) - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (180, 240, 255), 2)

    return frame


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
    speedometer = Speedometer(mapper, FPS, MPS_TO_MPH)
    model = YOLO(MODEL_PATH)
    
    # 初始化数据记录器
    data_recorder = VehicleDataRecorder(output_dir="results") if SAVE_VEHICLE_DATA else None
    video_recorder = AsyncVideoRecorder(
        output_dir=RECORD_OUTPUT_DIR,
        frame_size=(video_info.width, video_info.height),
        fps=FPS,
        filename_prefix=RECORD_OUTPUT_PREFIX,
        codec=RECORD_OUTPUT_CODEC,
        queue_size=RECORD_OUTPUT_QUEUE_SIZE,
        drop_frames=RECORD_OUTPUT_DROP_FRAMES,
        enabled=RECORD_OUTPUT_VIDEO,
    )

    print("="*60)
    print("车辆速度检测系统已启动")
    print("="*60)
    print("数据记录:")
    if SAVE_VEHICLE_DATA:
        print("  - 已启用：实时记录车辆ID、速度和轨迹坐标")
        print("  - 退出时自动保存到 results/ 目录")
    else:
        print("  - 已关闭：不保存车辆CSV/JSON/摘要")
    print("退出方式:")
    print("  - 在视频窗口按 'q' 键退出")
    print("  - 在视频窗口按 'ESC' 键退出")
    print("  - 关闭视频窗口退出")
    print("  - 视频播放完毕自动退出")
    if video_recorder.enabled:
        print(f"视频录制:")
        print(f"  - 已启用异步录制: {video_recorder.output_path}")
    else:
        print("视频录制:")
        print("  - 未启用或初始化失败")
    print("="*60)

    # 处理耗时统计（毫秒）
    processing_time_total_ms = 0.0
    processing_time_min_ms = float("inf")
    processing_time_max_ms = 0.0
    processed_frame_count = 0

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

        # 过滤无效 tracker_id，避免 None 导致遍历报错
        tracker_ids = detections.tracker_id
        if tracker_ids is None:
            valid_mask = np.zeros(len(detections), dtype=bool)
        else:
            valid_mask = np.array([tid is not None for tid in tracker_ids], dtype=bool)
        detections = detections[valid_mask]
        has_valid_tracker_ids = detections.tracker_id is not None and len(detections) > 0
        trace_ids = detections.tracker_id if has_valid_tracker_ids else []
        class_name_by_trace_id = {}
        if has_valid_tracker_ids:
            class_ids = detections.class_id
            model_names = getattr(model, "names", {})
            for idx, trace_id in enumerate(trace_ids):
                class_name = "unknown"
                if class_ids is not None and idx < len(class_ids) and class_ids[idx] is not None:
                    class_id = int(class_ids[idx])
                    if isinstance(model_names, dict):
                        class_name = model_names.get(class_id, "unknown")
                    elif isinstance(model_names, (list, tuple)) and 0 <= class_id < len(model_names):
                        class_name = model_names[class_id]
                class_name_by_trace_id[trace_id] = str(class_name) if class_name else "unknown"

        labels = []

        for trace_id in trace_ids:
            trace = annotators["trace"].trace.get(trace_id)
            speedometer.update_with_trace(trace_id, trace)
            speed = speedometer.get_current_speed(trace_id)
            class_name = class_name_by_trace_id.get(trace_id, "unknown")
            labels.append(f"{class_name} #Id:{trace_id} Speed:{speed} mile/h")
            
            # 记录车辆数据
            if trace is not None and len(trace) > 0:
                # 获取世界坐标轨迹
                try:
                    world_trace = mapper.map(trace)
                except Exception:
                    world_trace = None
                # 记录数据
                if data_recorder is not None:
                    data_recorder.record_vehicle(
                        vehicle_id=trace_id,
                        speed=speed,
                        image_trace=trace,
                        world_trace=world_trace,
                        class_name=class_name
                    )
        
        # 进入下一帧
        if data_recorder is not None:
            data_recorder.next_frame()

        # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # frame = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR)

        frame = annotators["bbox"].annotate(frame, detections)
        if has_valid_tracker_ids:
            frame = annotators["trace"].annotate(frame, detections)
            frame = annotators["label"].annotate(frame, detections, labels=labels)

        # 可视化监测区域（浅灰色透明阴影 + A/B/C/D 标注）
        if SHOW_MONITORING_AREA:
            frame = draw_monitoring_area_overlay(frame, IMAGE_POINTS)
        video_recorder.write(frame)

        cv.imshow("Vehicle Speed Estimation - YOLOv11", frame)
        
        # 计算处理时间，动态调整等待时间
        processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
        processing_time_total_ms += processing_time
        processing_time_min_ms = min(processing_time_min_ms, processing_time)
        processing_time_max_ms = max(processing_time_max_ms, processing_time)
        processed_frame_count += 1
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
    record_stats = video_recorder.close()
    cv.destroyAllWindows()
    
    # 保存数据
    if data_recorder is not None:
        print("\n正在保存车辆数据...")
    else:
        print("\n车辆数据保存开关已关闭，跳过保存。")
    try:
        stats = data_recorder.get_statistics() if data_recorder is not None else {
            "total_vehicles": 0,
            "total_records": 0,
            "total_frames": processed_frame_count
        }
        print(f"\n统计信息:")
        print(f"  - 检测到车辆数: {stats['total_vehicles']}")
        print(f"  - 总记录数: {stats['total_records']}")
        print(f"  - 总帧数: {stats['total_frames']}")
        if processed_frame_count > 0:
            avg_processing_time_ms = processing_time_total_ms / processed_frame_count
            print(f"  - 单帧处理耗时(平均): {avg_processing_time_ms:.2f} ms")
            print(f"  - 单帧处理耗时(最小): {processing_time_min_ms:.2f} ms")
            print(f"  - 单帧处理耗时(最大): {processing_time_max_ms:.2f} ms")
        if record_stats["enabled"]:
            print(f"  - 录制输出: {record_stats['output_path']}")
            print(f"  - 录制写入帧: {record_stats['frames_written']}")
            print(f"  - 录制丢帧: {record_stats['frames_dropped']}")
        
        if data_recorder is not None:
            data_recorder.save_all()
    except Exception as e:
        print(f"保存数据时出错: {e}")
    
    print("程序已正常退出")

if __name__ == "__main__":
    main()