import cv2 as cv
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time  # 添加这个导入
from collections import Counter, deque

from utils.downloader import download_video_if_needed
from utils.constants import IMAGE_POINTS, WORLD_POINTS
from config.settings import *

from modules.mapping import Cam2WorldMapper
from modules.speedometer import Speedometer
from modules.annotators import get_annotators
from modules.data_recorder import VehicleDataRecorder
from modules.lane_assigner import LaneAssigner
from modules.video_recorder import AsyncVideoRecorder
from zone.zone_trigger import create_zone


def draw_monitoring_area_overlay(
    frame,
    image_points,
    line_color=(170, 230, 255),
    line_thickness=3,
    label_text="Monitoring Area",
    label_font_scale=0.9,
    label_thickness=3,
    label_offset_x=0,
    label_offset_y=24,
    label_margin_x=8,
    label_margin_y=8
):
    """Draw ROI boundary and put a bold title below it."""
    points = np.array(image_points, dtype=np.int32)
    if points.shape[0] != 4:
        return frame

    cv.polylines(frame, [points], True, line_color, max(1, int(line_thickness)))

    frame_height, frame_width = frame.shape[:2]
    text_size, baseline = cv.getTextSize(
        str(label_text),
        cv.FONT_HERSHEY_SIMPLEX,
        float(label_font_scale),
        max(1, int(label_thickness))
    )
    text_width, text_height = text_size

    min_x = int(np.min(points[:, 0]))
    max_y = int(np.max(points[:, 1]))
    preferred_x = min_x + int(label_offset_x)
    preferred_y = max_y + int(label_offset_y) + text_height

    max_x = max(int(label_margin_x), frame_width - text_width - int(label_margin_x))
    label_x = min(max(preferred_x, int(label_margin_x)), max_x)

    min_baseline_y = int(label_margin_y) + text_height
    max_baseline_y = frame_height - int(label_margin_y) - baseline
    label_y = min(max(preferred_y, min_baseline_y), max_baseline_y)

    cv.putText(
        frame,
        str(label_text),
        (label_x, label_y),
        cv.FONT_HERSHEY_SIMPLEX,
        float(label_font_scale),
        line_color,
        max(1, int(label_thickness))
    )

    return frame


def draw_lane_boundary_lines_overlay(frame, lane_lines, line_color=(0, 0, 255), line_thickness=1):
    """
    Draw lane boundary lines without per-line labels.

    Args:
        frame: current frame.
        lane_lines: boundary line list, each item is [(x1, y1), (x2, y2)].
        line_color: BGR color tuple.
        line_thickness: line thickness.
    """
    if not lane_lines:
        return frame

    for line in lane_lines:
        if line is None or len(line) != 2:
            continue
        p1, p2 = line[0], line[1]
        if p1 is None or p2 is None or len(p1) != 2 or len(p2) != 2:
            continue

        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        cv.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)

    return frame


def draw_status_legend_overlay(
    frame,
    normal_color=(128, 0, 0),
    abnormal_color=(0, 0, 255),
    normal_text="Normal",
    abnormal_text="Abnormal",
    margin_x=20,
    margin_y=20,
    box_size=14,
    item_gap=10,
    text_gap=8,
    font=cv.FONT_HERSHEY_DUPLEX,
    font_scale=0.6,
    text_thickness=2,
    text_color=(255, 255, 255),
    bg_color=(255, 255, 255),
    border_color=(255, 255, 255),
    border_thickness=3,
    padding_x=10,
    padding_y=8,
):
    """Draw status legend at bottom-left corner."""
    frame_height = frame.shape[0]
    normal_baseline_y = frame_height - int(margin_y) - int(box_size) - int(item_gap)
    abnormal_baseline_y = frame_height - int(margin_y)
    items = [
        (normal_color, str(normal_text), normal_baseline_y),
        (abnormal_color, str(abnormal_text), abnormal_baseline_y),
    ]

    legend_left = int(margin_x)
    legend_top = frame_height
    legend_right = int(margin_x + box_size)
    legend_bottom = 0

    for _, text, baseline_y in items:
        text_size, text_baseline = cv.getTextSize(
            text,
            font,
            float(font_scale),
            max(1, int(text_thickness))
        )
        text_width, text_height = text_size
        box_center_y = baseline_y - box_size / 2
        text_visual_top = int(box_center_y - text_height / 2)
        text_visual_bottom = int(box_center_y + text_height / 2 + text_baseline)
        item_top = min(int(baseline_y - box_size), text_visual_top)
        item_bottom = max(int(baseline_y), text_visual_bottom)
        item_right = int(margin_x + box_size + text_gap + text_width)

        legend_top = min(legend_top, item_top)
        legend_bottom = max(legend_bottom, item_bottom)
        legend_right = max(legend_right, item_right)

    border_top_left = (max(0, legend_left - int(padding_x)), max(0, legend_top - int(padding_y)))
    border_bottom_right = (legend_right + int(padding_x), legend_bottom + int(padding_y))
    
    # 绘制纯色背景填充
    cv.rectangle(frame, border_top_left, border_bottom_right, bg_color, -1)
    
    # 绘制外边框
    cv.rectangle(
        frame,
        border_top_left,
        border_bottom_right,
        border_color,
        max(1, int(border_thickness))
    )

    for color, text, baseline_y in items:
        box_top_left = (int(margin_x), int(baseline_y - box_size))
        box_bottom_right = (int(margin_x + box_size), int(baseline_y))
        cv.rectangle(frame, box_top_left, box_bottom_right, color, -1)

        text_size, _ = cv.getTextSize(
            text, font,
            float(font_scale), max(1, int(text_thickness))
        )
        text_height = text_size[1]
        box_center_y = baseline_y - box_size / 2
        text_origin = (int(margin_x + box_size + text_gap), int(box_center_y + text_height / 2))

        cv.putText(
            frame,
            text,
            text_origin,
            font,
            float(font_scale),
            text_color,
            max(1, int(text_thickness))
        )
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
    lane_assignment_mode = globals().get("LANE_ASSIGNMENT_MODE", "x_threshold")
    lane_boundary_lines = globals().get("LANE_BOUNDARY_LINES", [])
    lane_boundaries_x = globals().get("LANE_BOUNDARIES_X", [])
    emergency_lane_enabled = globals().get("EMERGENCY_LANE_ENABLED", False)
    emergency_lane_rules = globals().get("EMERGENCY_LANE_RULES", [])
    emergency_lane_between_boundaries = globals().get("EMERGENCY_LANE_BETWEEN_BOUNDARIES", None)
    show_lane_boundary_lines = globals().get("SHOW_LANE_BOUNDARY_LINES", False)
    lane_line_color = tuple(globals().get("LANE_LINE_COLOR", (0, 0, 255)))
    lane_line_thickness = int(globals().get("LANE_LINE_THICKNESS", 1))
    monitoring_area_line_color = tuple(globals().get("MONITORING_AREA_LINE_COLOR", (170, 230, 255)))
    monitoring_area_line_thickness = int(globals().get("MONITORING_AREA_LINE_THICKNESS", 3))
    monitoring_area_label_text = str(globals().get("MONITORING_AREA_LABEL_TEXT", "Monitoring Area"))
    monitoring_area_label_font_scale = float(globals().get("MONITORING_AREA_LABEL_FONT_SCALE", 0.9))
    monitoring_area_label_thickness = int(globals().get("MONITORING_AREA_LABEL_THICKNESS", 3))
    monitoring_area_label_offset_x = int(globals().get("MONITORING_AREA_LABEL_OFFSET_X", 0))
    monitoring_area_label_offset_y = int(globals().get("MONITORING_AREA_LABEL_OFFSET_Y", 24))
    monitoring_area_label_margin_x = int(globals().get("MONITORING_AREA_LABEL_MARGIN_X", 8))
    monitoring_area_label_margin_y = int(globals().get("MONITORING_AREA_LABEL_MARGIN_Y", 8))
    speed_limit_mph = float(globals().get("SPEED_LIMIT_MPH", 65))
    max_valid_speed_mph = float(globals().get("MAX_VALID_SPEED_MPH", 95))
    normal_bbox_trace_color = tuple(globals().get("NORMAL_BBOX_TRACE_COLOR", (128, 0, 0)))
    abnormal_bbox_trace_color = tuple(globals().get("ABNORMAL_BBOX_TRACE_COLOR", (0, 0, 255)))
    abnormal_bbox_thickness = int(globals().get("ABNORMAL_BBOX_THICKNESS", 2))
    abnormal_trace_thickness = int(globals().get("ABNORMAL_TRACE_THICKNESS", 2))
    show_status_legend = bool(globals().get("SHOW_STATUS_LEGEND", True))
    status_legend_margin_x = int(globals().get("STATUS_LEGEND_MARGIN_X", 20))
    status_legend_margin_y = int(globals().get("STATUS_LEGEND_MARGIN_Y", 20))
    status_legend_box_size = int(globals().get("STATUS_LEGEND_BOX_SIZE", 14))
    status_legend_item_gap = int(globals().get("STATUS_LEGEND_ITEM_GAP", 10))
    status_legend_text_gap = int(globals().get("STATUS_LEGEND_TEXT_GAP", 8))
    _font_name = str(globals().get("STATUS_LEGEND_FONT", "FONT_HERSHEY_DUPLEX"))
    status_legend_font = getattr(cv, _font_name, cv.FONT_HERSHEY_DUPLEX)
    status_legend_font_scale = float(globals().get("STATUS_LEGEND_FONT_SCALE", 0.6))
    status_legend_thickness = int(globals().get("STATUS_LEGEND_THICKNESS", 2))
    status_legend_text_color = tuple(globals().get("STATUS_LEGEND_TEXT_COLOR", (0, 0, 0)))
    status_legend_bg_color = tuple(globals().get("STATUS_LEGEND_BG_COLOR", (255, 255, 255)))
    status_legend_normal_text = str(globals().get("STATUS_LEGEND_NORMAL_TEXT", "Normal"))
    status_legend_abnormal_text = str(globals().get("STATUS_LEGEND_ABNORMAL_TEXT", "Abnormal"))
    status_legend_border_color = tuple(globals().get("STATUS_LEGEND_BORDER_COLOR", (255, 255, 255)))
    status_legend_border_thickness = int(globals().get("STATUS_LEGEND_BORDER_THICKNESS", 3))
    status_legend_padding_x = int(globals().get("STATUS_LEGEND_PADDING_X", 10))
    status_legend_padding_y = int(globals().get("STATUS_LEGEND_PADDING_Y", 8))
    class_smoothing_enabled = bool(globals().get("CLASS_SMOOTHING_ENABLED", True))
    class_smoothing_window = max(1, int(globals().get("CLASS_SMOOTHING_WINDOW", 10)))
    class_smoothing_min_obs = max(1, int(globals().get("CLASS_SMOOTHING_MIN_OBS", 3)))
    lane_assigner = LaneAssigner(
        lane_boundaries_x=lane_boundaries_x,
        lane_boundary_lines=lane_boundary_lines,
        mode=lane_assignment_mode,
        emergency_lane_enabled=emergency_lane_enabled,
        emergency_lane_between_boundaries=emergency_lane_between_boundaries,
        emergency_lane_rules=emergency_lane_rules
    ) if ENABLE_LANE_ASSIGNMENT else None
    
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
    frame_index = 0
    class_history_by_trace_id = {}
    last_seen_frame_by_trace_id = {}

    while True:
        start_time = time.time()  # 记录开始时间
        
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

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
                raw_class_name = "unknown"
                if class_ids is not None and idx < len(class_ids) and class_ids[idx] is not None:
                    class_id = int(class_ids[idx])
                    if isinstance(model_names, dict):
                        raw_class_name = model_names.get(class_id, "unknown")
                    elif isinstance(model_names, (list, tuple)) and 0 <= class_id < len(model_names):
                        raw_class_name = model_names[class_id]
                raw_class_name = str(raw_class_name) if raw_class_name else "unknown"

                if class_smoothing_enabled:
                    history = class_history_by_trace_id.get(trace_id)
                    if history is None:
                        history = deque(maxlen=class_smoothing_window)
                        class_history_by_trace_id[trace_id] = history
                    history.append(raw_class_name)
                    last_seen_frame_by_trace_id[trace_id] = frame_index

                    if len(history) >= class_smoothing_min_obs:
                        vote_counter = Counter(history)
                        top_vote_count = max(vote_counter.values())
                        top_labels = {label for label, count in vote_counter.items() if count == top_vote_count}
                        if len(top_labels) == 1:
                            stable_class_name = next(iter(top_labels))
                        else:
                            stable_class_name = raw_class_name
                            for label in reversed(history):
                                if label in top_labels:
                                    stable_class_name = label
                                    break
                    else:
                        stable_class_name = raw_class_name
                    class_name_by_trace_id[trace_id] = stable_class_name
                else:
                    class_name_by_trace_id[trace_id] = raw_class_name

        if class_smoothing_enabled and class_history_by_trace_id:
            stale_track_gap = max(30, class_smoothing_window * 3)
            stale_trace_ids = [
                tid for tid, seen_frame in last_seen_frame_by_trace_id.items()
                if frame_index - seen_frame > stale_track_gap
            ]
            for stale_trace_id in stale_trace_ids:
                class_history_by_trace_id.pop(stale_trace_id, None)
                last_seen_frame_by_trace_id.pop(stale_trace_id, None)

        labels = []
        abnormal_trace_ids = set()

        for idx, trace_id in enumerate(trace_ids):
            trace = annotators["trace"].trace.get(trace_id)
            speedometer.update_with_trace(trace_id, trace)
            speed = speedometer.get_current_speed(trace_id)
            is_speeding = speed > speed_limit_mph
            is_outlier = speed > max_valid_speed_mph
            class_name = class_name_by_trace_id.get(trace_id, "unknown")
            lane_id = None
            is_emergency_lane = False
            if lane_assigner is not None and trace is not None and len(trace) > 0:
                lane_id = lane_assigner.get_lane_id_from_point(trace[-1])
                is_emergency_lane = lane_assigner.is_emergency_lane(lane_id)
            is_abnormal = is_speeding or is_emergency_lane
            if is_abnormal:
                abnormal_trace_ids.add(trace_id)
            if is_speeding and is_emergency_lane:
                abnormal_status = "Speeding + EmergencyLane"
            elif is_speeding:
                abnormal_status = "Speeding"
            elif is_emergency_lane:
                abnormal_status = "EmergencyLane"
            else:
                abnormal_status = "Normal"

            if is_abnormal:
                labels.append(
                    f"Type: {class_name}\n"
                    f"Speed: {speed:.1f} mph\n"
                    f"Status: {abnormal_status}"
                )
            else:
                labels.append(
                    f"Type: {class_name}\n"
                    f"Speed: {speed:.1f} mph"
                )
            
            # 记录车辆数据
            if trace is not None and len(trace) > 0:
                # 获取世界坐标轨迹
                try:
                    world_trace = mapper.map(trace)
                except Exception:
                    world_trace = None
                # 记录数据
                if data_recorder is not None and not is_outlier:
                    data_recorder.record_vehicle(
                        vehicle_id=trace_id,
                        speed=speed,
                        image_trace=trace,
                        world_trace=world_trace,
                        class_name=class_name,
                        lane_id=lane_id,
                        is_emergency_lane=is_emergency_lane,
                        is_speeding=is_speeding
                    )
        
        # 进入下一帧
        if data_recorder is not None:
            data_recorder.next_frame()

        # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # frame = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR)

        # 先绘制监测区域，再绘制车辆框/轨迹/标签，避免 ROI 覆盖车辆标签
        if SHOW_MONITORING_AREA:
            frame = draw_monitoring_area_overlay(
                frame=frame,
                image_points=IMAGE_POINTS,
                line_color=monitoring_area_line_color,
                line_thickness=monitoring_area_line_thickness,
                label_text=monitoring_area_label_text,
                label_font_scale=monitoring_area_label_font_scale,
                label_thickness=monitoring_area_label_thickness,
                label_offset_x=monitoring_area_label_offset_x,
                label_offset_y=monitoring_area_label_offset_y,
                label_margin_x=monitoring_area_label_margin_x,
                label_margin_y=monitoring_area_label_margin_y
            )

        frame = annotators["bbox"].annotate(frame, detections)
        if has_valid_tracker_ids and len(detections) > 0:
            for idx, trace_id in enumerate(trace_ids):
                if trace_id in abnormal_trace_ids and idx < len(detections.xyxy):
                    x1, y1, x2, y2 = detections.xyxy[idx]
                    cv.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        abnormal_bbox_trace_color,
                        abnormal_bbox_thickness
                    )
        if has_valid_tracker_ids:
            frame = annotators["trace"].annotate(frame, detections)
            for trace_id in abnormal_trace_ids:
                trace_points = annotators["trace"].trace.get(trace_id)
                if trace_points is None or len(trace_points) < 2:
                    continue
                polyline_points = np.array(trace_points, dtype=np.int32).reshape(-1, 1, 2)
                cv.polylines(
                    frame,
                    [polyline_points],
                    False,
                    abnormal_bbox_trace_color,
                    abnormal_trace_thickness
                )

            abnormal_mask = np.array([tid in abnormal_trace_ids for tid in trace_ids], dtype=bool)
            abnormal_detections = detections[abnormal_mask]
            normal_detections = detections[~abnormal_mask]
            abnormal_labels = [label for i, label in enumerate(labels) if abnormal_mask[i]]
            normal_labels = [label for i, label in enumerate(labels) if not abnormal_mask[i]]

            if len(abnormal_detections) > 0:
                frame = annotators["abnormal_label"].annotate(frame, abnormal_detections, labels=abnormal_labels)
            if len(normal_detections) > 0:
                frame = annotators["label"].annotate(frame, normal_detections, labels=normal_labels)

        if show_lane_boundary_lines:
            frame = draw_lane_boundary_lines_overlay(
                frame=frame,
                lane_lines=lane_boundary_lines,
                line_color=lane_line_color,
                line_thickness=lane_line_thickness
            )

        if show_status_legend:
            frame = draw_status_legend_overlay(
                frame=frame,
                normal_color=normal_bbox_trace_color,
                abnormal_color=abnormal_bbox_trace_color,
                normal_text=status_legend_normal_text,
                abnormal_text=status_legend_abnormal_text,
                margin_x=status_legend_margin_x,
                margin_y=status_legend_margin_y,
                box_size=status_legend_box_size,
                item_gap=status_legend_item_gap,
                text_gap=status_legend_text_gap,
                font=status_legend_font,
                font_scale=status_legend_font_scale,
                text_thickness=status_legend_thickness,
                text_color=status_legend_text_color,
                bg_color=status_legend_bg_color,
                border_color=status_legend_border_color,
                border_thickness=status_legend_border_thickness,
                padding_x=status_legend_padding_x,
                padding_y=status_legend_padding_y
            )
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