"""
交互式标注工具 - 用于在视频上选择 IMAGE_POINTS
使用方法: python utils/annotate_points.py
"""
import cv2 as cv
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.downloader import download_video_if_needed
from config.settings import VIDEO_PATH

class PointAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        self.points = []
        self.current_frame = None
        self.frame_number = 0
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"点 {len(self.points)}: ({x}, {y})")
                
                # 在图像上绘制点
                cv.circle(self.current_frame, (x, y), 8, (0, 255, 0), -1)
                cv.circle(self.current_frame, (x, y), 12, (0, 255, 0), 2)
                
                # 添加点编号
                cv.putText(self.current_frame, str(len(self.points)), 
                          (x + 15, y - 15), cv.FONT_HERSHEY_SIMPLEX, 
                          0.8, (0, 255, 0), 2)
                
                # 如果选择了4个点，绘制多边形
                if len(self.points) == 4:
                    pts = np.array(self.points, np.int32)
                    cv.polylines(self.current_frame, [pts], True, (0, 255, 0), 2)
                    # 填充半透明区域
                    overlay = self.current_frame.copy()
                    cv.fillPoly(overlay, [pts], (0, 255, 0))
                    cv.addWeighted(overlay, 0.3, self.current_frame, 0.7, 0, self.current_frame)
                    print("\n✓ 已选择4个点！")
                    print("按 's' 保存，按 'r' 重新选择，按 'q' 退出")
                
                cv.imshow("标注工具 - 选择4个点", self.current_frame)
        
        elif event == cv.EVENT_MOUSEMOVE:
            # 实时显示鼠标位置
            temp_frame = self.current_frame.copy()
            cv.circle(temp_frame, (x, y), 5, (255, 0, 0), 1)
            cv.putText(temp_frame, f"({x}, {y})", (x + 10, y - 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 如果已有点，绘制连线预览
            if len(self.points) > 0:
                for i, pt in enumerate(self.points):
                    cv.circle(temp_frame, pt, 8, (0, 255, 0), -1)
                    cv.putText(temp_frame, str(i + 1), (pt[0] + 15, pt[1] - 15),
                              cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if len(self.points) >= 2:
                    pts = np.array(self.points + [(x, y)], np.int32)
                    cv.polylines(temp_frame, [pts], False, (0, 255, 255), 1)
            
            cv.imshow("标注工具 - 选择4个点", temp_frame)
    
    def load_frame(self, frame_num=None):
        """加载指定帧"""
        if frame_num is not None:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
            self.frame_number = frame_num
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            self.frame_number = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))
            return True
        return False
    
    def reset_points(self):
        """重置选择的点"""
        self.points = []
        self.load_frame(self.frame_number)
        print("\n已重置，请重新选择点")
    
    def save_points(self):
        """保存选择的点"""
        if len(self.points) != 4:
            print("错误：需要选择4个点！")
            return False
        
        # 输出到控制台
        print("\n" + "="*50)
        print("选择的 IMAGE_POINTS:")
        print("="*50)
        print(f"IMAGE_POINTS = {self.points}")
        print("\n复制以下代码到 constants.py:")
        print("-"*50)
        print(f"IMAGE_POINTS = {self.points}")
        print("-"*50)
        
        # 保存到文件
        output_file = "annotated_points.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 标注的 IMAGE_POINTS\n")
            f.write(f"# 视频: {self.video_path}\n")
            f.write(f"# 帧号: {self.frame_number}\n")
            f.write(f"# 分辨率: {self.current_frame.shape[1]}x{self.current_frame.shape[0]}\n")
            f.write(f"\nIMAGE_POINTS = {self.points}\n")
        
        print(f"\n✓ 已保存到文件: {output_file}")
        print("="*50)
        
        return True
    
    def run(self):
        """运行标注工具"""
        # 加载第一帧
        if not self.load_frame(0):
            print("错误：无法读取视频")
            return
        
        print("="*50)
        print("交互式标注工具")
        print("="*50)
        print(f"视频: {self.video_path}")
        print(f"分辨率: {int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))}")
        print(f"总帧数: {self.total_frames}")
        print(f"FPS: {self.fps:.2f}")
        print("\n操作说明:")
        print("  - 鼠标左键点击: 选择点（需要选择4个点）")
        print("  - 按 's': 保存选择的点")
        print("  - 按 'r': 重新选择点")
        print("  - 按 'n': 下一帧")
        print("  - 按 'p': 上一帧")
        print("  - 按 'f': 跳转到指定帧")
        print("  - 按 'q' 或 ESC: 退出")
        print("\n请按顺序点击4个点：")
        print("  1. 左上角")
        print("  2. 右上角")
        print("  3. 右下角")
        print("  4. 左下角")
        print("="*50)
        
        cv.namedWindow("标注工具 - 选择4个点", cv.WINDOW_NORMAL)
        cv.setMouseCallback("标注工具 - 选择4个点", self.mouse_callback)
        
        while True:
            cv.imshow("标注工具 - 选择4个点", self.current_frame)
            
            key = cv.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' 或 ESC
                break
            elif key == ord('s'):  # 保存
                self.save_points()
            elif key == ord('r'):  # 重置
                self.reset_points()
            elif key == ord('n'):  # 下一帧
                if self.frame_number < self.total_frames - 1:
                    self.frame_number += 1
                    self.load_frame(self.frame_number)
                    self.points = []
                    print(f"\n已切换到第 {self.frame_number} 帧")
            elif key == ord('p'):  # 上一帧
                if self.frame_number > 0:
                    self.frame_number -= 1
                    self.load_frame(self.frame_number)
                    self.points = []
                    print(f"\n已切换到第 {self.frame_number} 帧")
            elif key == ord('f'):  # 跳转到指定帧
                try:
                    frame_num = int(input(f"\n请输入帧号 (0-{self.total_frames-1}): "))
                    if 0 <= frame_num < self.total_frames:
                        self.frame_number = frame_num
                        self.load_frame(self.frame_number)
                        self.points = []
                        print(f"已跳转到第 {self.frame_number} 帧")
                    else:
                        print("帧号超出范围！")
                except ValueError:
                    print("请输入有效的数字！")
        
        self.cap.release()
        cv.destroyAllWindows()
        print("\n标注工具已关闭")


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='交互式标注工具 - 选择视频上的 IMAGE_POINTS')
    parser.add_argument('--video', '-v', type=str, default=None,
                        help='视频文件路径（可选，如果不指定则使用 config/settings.py 中的 VIDEO_PATH）')
    args = parser.parse_args()
    
    # 获取视频路径
    if args.video:
        # 使用命令行指定的视频路径
        video_path = args.video
        if not os.path.isabs(video_path):
            # 如果是相对路径，转换为绝对路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_path = os.path.join(base_dir, video_path)
    else:
        # 使用配置文件中的视频路径
        try:
            video_path = download_video_if_needed()
        except Exception as e:
            print(f"错误：无法获取视频路径: {e}")
            print("\n提示：你可以使用 --video 参数直接指定视频路径")
            print("例如: python annotate.py --video path/to/your/video.mp4")
            return
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        print(f"\n当前配置的视频路径: {video_path}")
        print("请检查 config/settings.py 中的 VIDEO_PATH 设置")
        print("或者使用 --video 参数指定视频路径")
        return
    
    # 创建标注工具并运行
    annotator = PointAnnotator(video_path)
    annotator.run()


if __name__ == "__main__":
    main()
