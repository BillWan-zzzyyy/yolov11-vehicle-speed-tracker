"""
交互式标注工具 - 支持 ROI 四点标注与车道边界线标注。
使用方法:
    python utils/annotate_points.py --mode roi
    python utils/annotate_points.py --mode lane
"""
import cv2 as cv
import numpy as np
import sys
import os
import re

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.downloader import download_video_if_needed

WINDOW_NAME = "标注工具"


class PointAnnotator:
    """
    交互式标注器。

    Args:
        video_path: 视频路径。
        annotate_mode: 标注模式，可选 "roi" 或 "lane"。
    """

    def __init__(self, video_path, annotate_mode="lane"):
        self.video_path = video_path
        self.annotate_mode = annotate_mode
        self.cap = cv.VideoCapture(video_path)
        self.roi_points = []
        self.lane_lines = []
        self.pending_lane_point = None
        self.base_frame = None
        self.current_frame = None
        self.frame_number = 0
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv.CAP_PROP_FPS)

    def _replace_or_append_setting(self, content, key, value_repr):
        pattern = rf"^{key}\s*=.*$"
        replacement = f"{key} = {value_repr}"
        if re.search(pattern, content, flags=re.MULTILINE):
            return re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
        if not content.endswith("\n"):
            content += "\n"
        return f"{content}{replacement}\n"

    def _format_lane_lines(self):
        formatted_lines = []
        for line in self.lane_lines:
            (x1, y1), (x2, y2) = line
            formatted_lines.append(f"[({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})]")
        return "[" + ", ".join(formatted_lines) + "]"

    def _update_settings_with_lane_config(self):
        settings_path = os.path.join(PROJECT_ROOT, "config", "settings.py")
        if not os.path.exists(settings_path):
            print(f"警告：未找到 settings.py，跳过自动写入：{settings_path}")
            return False

        with open(settings_path, "r", encoding="utf-8") as file:
            content = file.read()

        content = self._replace_or_append_setting(content, "ENABLE_LANE_ASSIGNMENT", "True")
        content = self._replace_or_append_setting(content, "LANE_ASSIGNMENT_MODE", '"line_segments"')
        content = self._replace_or_append_setting(content, "LANE_BOUNDARY_LINES", self._format_lane_lines())

        with open(settings_path, "w", encoding="utf-8") as file:
            file.write(content)
        return True

    def _draw_roi_annotations(self, canvas, cursor_point=None):
        for idx, point in enumerate(self.roi_points):
            cv.circle(canvas, point, 8, (0, 255, 0), -1)
            cv.putText(canvas, str(idx + 1), (point[0] + 10, point[1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(self.roi_points) == 4:
            pts = np.array(self.roi_points, dtype=np.int32)
            cv.polylines(canvas, [pts], True, (0, 255, 0), 2)
            overlay = canvas.copy()
            cv.fillPoly(overlay, [pts], (0, 255, 0))
            cv.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        elif len(self.roi_points) > 0 and cursor_point is not None:
            preview = self.roi_points + [cursor_point]
            pts = np.array(preview, dtype=np.int32)
            cv.polylines(canvas, [pts], False, (0, 255, 255), 1)

    def _draw_lane_annotations(self, canvas, cursor_point=None):
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.lane_lines, start=1):
            cv.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text_anchor = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv.putText(canvas, f"LaneBoundary{idx}", (text_anchor[0] + 8, text_anchor[1] - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if self.pending_lane_point is not None:
            cv.circle(canvas, self.pending_lane_point, 8, (0, 165, 255), -1)
            cv.putText(canvas, "start", (self.pending_lane_point[0] + 8, self.pending_lane_point[1] - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if cursor_point is not None:
                cv.line(canvas, self.pending_lane_point, cursor_point, (255, 200, 0), 1)

    def _render_frame(self, cursor_point=None):
        if self.base_frame is None:
            return
        canvas = self.base_frame.copy()
        if self.annotate_mode == "roi":
            self._draw_roi_annotations(canvas, cursor_point=cursor_point)
        else:
            self._draw_lane_annotations(canvas, cursor_point=cursor_point)
        self.current_frame = canvas
        cv.imshow(WINDOW_NAME, self.current_frame)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数。"""
        if self.base_frame is None:
            return

        if event == cv.EVENT_LBUTTONDOWN:
            if self.annotate_mode == "roi":
                if len(self.roi_points) < 4:
                    self.roi_points.append((x, y))
                    print(f"ROI 点 {len(self.roi_points)}: ({x}, {y})")
                    if len(self.roi_points) == 4:
                        print("\n✓ ROI 已选择4个点，可按 's' 保存。")
                else:
                    print("ROI 已满4点，按 'r' 重置或按 'u' 撤销。")
            else:
                if self.pending_lane_point is None:
                    self.pending_lane_point = (x, y)
                    print(f"车道线起点: ({x}, {y})")
                else:
                    lane_line = (self.pending_lane_point, (x, y))
                    self.lane_lines.append(lane_line)
                    self.pending_lane_point = None
                    print(f"车道线 {len(self.lane_lines)}: {lane_line}")
            self._render_frame()

        elif event == cv.EVENT_MOUSEMOVE:
            self._render_frame(cursor_point=(x, y))

    def load_frame(self, frame_num=None):
        """
        加载指定帧并刷新画布。

        Args:
            frame_num: 目标帧号，None 表示读取当前位置。
        """
        if frame_num is not None:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
            self.frame_number = frame_num

        ret, frame = self.cap.read()
        if not ret:
            return False
        self.base_frame = frame.copy()
        self.current_frame = frame.copy()
        self.frame_number = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))
        self._render_frame()
        return True

    def reset_annotations(self):
        """重置当前模式下的标注。"""
        if self.annotate_mode == "roi":
            self.roi_points = []
        else:
            self.lane_lines = []
            self.pending_lane_point = None
        self._render_frame()
        print("已重置当前标注。")

    def undo_last(self):
        """撤销最近一次标注动作。"""
        if self.annotate_mode == "roi":
            if self.roi_points:
                removed = self.roi_points.pop()
                print(f"已撤销 ROI 点: {removed}")
            else:
                print("没有可撤销的 ROI 点。")
        else:
            if self.pending_lane_point is not None:
                print(f"已撤销车道线起点: {self.pending_lane_point}")
                self.pending_lane_point = None
            elif self.lane_lines:
                removed = self.lane_lines.pop()
                print(f"已撤销车道线: {removed}")
            else:
                print("没有可撤销的车道线。")
        self._render_frame()

    def save_annotations(self):
        """
        保存当前模式的标注结果。

        ROI 模式:
            输出 IMAGE_POINTS 片段到文本文件。
        车道模式:
            输出 LANE_BOUNDARY_LINES 并自动更新 settings.py。
        """
        if self.annotate_mode == "roi":
            if len(self.roi_points) != 4:
                print("错误：ROI 模式需要恰好 4 个点。")
                return False
            output_file = "annotated_points.txt"
            with open(output_file, "w", encoding="utf-8") as file:
                file.write("# 标注的 IMAGE_POINTS\n")
                file.write(f"# 视频: {self.video_path}\n")
                file.write(f"# 帧号: {self.frame_number}\n")
                file.write(f"# 分辨率: {self.base_frame.shape[1]}x{self.base_frame.shape[0]}\n")
                file.write(f"\nIMAGE_POINTS = {self.roi_points}\n")
            print("\n" + "=" * 50)
            print("ROI 标注结果:")
            print(f"IMAGE_POINTS = {self.roi_points}")
            print(f"已保存到: {output_file}")
            print("=" * 50)
            return True

        if self.pending_lane_point is not None:
            print("错误：存在未闭合车道线，请再点击一个终点或按 'u' 撤销。")
            return False
        if not self.lane_lines:
            print("错误：至少需要标注 1 条车道边界线。")
            return False

        lane_repr = self._format_lane_lines()
        output_file = "annotated_lane_lines.txt"
        with open(output_file, "w", encoding="utf-8") as file:
            file.write("# 标注的车道边界线（从左到右）\n")
            file.write(f"# 视频: {self.video_path}\n")
            file.write(f"# 帧号: {self.frame_number}\n")
            file.write(f"# 分辨率: {self.base_frame.shape[1]}x{self.base_frame.shape[0]}\n")
            file.write('\nLANE_ASSIGNMENT_MODE = "line_segments"\n')
            file.write(f"LANE_BOUNDARY_LINES = {lane_repr}\n")

        updated = self._update_settings_with_lane_config()
        print("\n" + "=" * 60)
        print("车道线标注结果:")
        print('LANE_ASSIGNMENT_MODE = "line_segments"')
        print(f"LANE_BOUNDARY_LINES = {lane_repr}")
        print(f"已保存到: {output_file}")
        if updated:
            print("已自动写入 config/settings.py。")
        print("=" * 60)
        return True

    def _print_help(self):
        mode_desc = "ROI四点标注" if self.annotate_mode == "roi" else "车道线标注"
        print("=" * 60)
        print(f"交互式标注工具 - {mode_desc}")
        print("=" * 60)
        print(f"视频: {self.video_path}")
        print(f"分辨率: {int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))}")
        print(f"总帧数: {self.total_frames}")
        print(f"FPS: {self.fps:.2f}")
        print("\n操作说明:")
        print("  - 鼠标左键: 标注点")
        print("  - s: 保存")
        print("  - u: 撤销最近一次标注")
        print("  - r: 重置当前模式标注")
        print("  - n/p: 下一帧/上一帧（切换帧会保留当前标注）")
        print("  - f: 跳转到指定帧")
        print("  - m: 切换标注模式（roi/lane）")
        print("  - q 或 ESC: 退出")
        print("=" * 60)

    def run(self):
        """运行标注工具。"""
        if not self.load_frame(0):
            print("错误：无法读取视频。")
            return

        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
        cv.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        self._print_help()

        while True:
            if self.current_frame is not None:
                cv.imshow(WINDOW_NAME, self.current_frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                self.save_annotations()
            elif key == ord("u"):
                self.undo_last()
            elif key == ord("r"):
                self.reset_annotations()
            elif key == ord("m"):
                self.annotate_mode = "lane" if self.annotate_mode == "roi" else "roi"
                print(f"已切换模式: {self.annotate_mode}")
                self._print_help()
                self._render_frame()
            elif key == ord("n"):
                if self.frame_number < self.total_frames - 1:
                    self.load_frame(self.frame_number + 1)
                    print(f"已切换到第 {self.frame_number} 帧。")
            elif key == ord("p"):
                if self.frame_number > 1:
                    self.load_frame(self.frame_number - 1)
                    print(f"已切换到第 {self.frame_number} 帧。")
            elif key == ord("f"):
                try:
                    frame_num = int(input(f"请输入帧号 (0-{self.total_frames - 1}): "))
                    if 0 <= frame_num < self.total_frames:
                        self.load_frame(frame_num)
                        print(f"已跳转到第 {self.frame_number} 帧。")
                    else:
                        print("帧号超出范围。")
                except ValueError:
                    print("请输入有效数字。")

        self.cap.release()
        cv.destroyAllWindows()
        print("标注工具已关闭。")


def main():
    """主函数。"""
    import argparse

    parser = argparse.ArgumentParser(description="交互式标注工具（ROI/车道线）")
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="视频路径（可选，不传则使用 settings.py 默认下载逻辑）",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="lane",
        choices=["roi", "lane"],
        help="标注模式：roi 或 lane（默认 lane）",
    )
    args = parser.parse_args()

    if args.video:
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.join(PROJECT_ROOT, video_path)
    else:
        try:
            video_path = download_video_if_needed()
        except Exception as err:
            print(f"错误：无法获取视频路径: {err}")
            print("提示：可用 --video 指定本地视频。")
            return

    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return

    annotator = PointAnnotator(video_path=video_path, annotate_mode=args.mode)
    annotator.run()


if __name__ == "__main__":
    main()
