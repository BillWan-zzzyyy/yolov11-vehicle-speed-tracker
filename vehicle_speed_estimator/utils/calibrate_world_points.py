"""
WORLD_POINTS 校准工具
使用方法: python utils/calibrate_world_points.py
通过选择参考物体（如车道宽度）来自动计算 WORLD_POINTS
"""
import cv2 as cv
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.downloader import download_video_if_needed
from utils.constants import IMAGE_POINTS

class WorldPointsCalibrator:
    def __init__(self, video_path, image_points):
        self.video_path = video_path
        self.image_points = image_points
        self.cap = cv.VideoCapture(video_path)
        self.width_reference_points = []  # 宽度方向参考物体的两个端点
        self.height_reference_points = []  # 高度方向参考物体的两个端点
        self.width_length_meters = None
        self.height_length_meters = None
        self.current_frame = None
        self.frame_number = 0
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.width_scale = None  # 宽度方向的转换比例（米/像素）
        self.height_scale = None  # 高度方向的转换比例（米/像素）
        self.calibration_mode = "width"  # "width" 或 "height"
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数 - 选择参考物体"""
        if event == cv.EVENT_LBUTTONDOWN:
            if self.calibration_mode == "width":
                ref_points = self.width_reference_points
                color = (255, 0, 0)  # 蓝色表示宽度
            else:
                ref_points = self.height_reference_points
                color = (0, 0, 255)  # 红色表示高度
            
            if len(ref_points) < 2:
                ref_points.append((x, y))
                mode_name = "宽度" if self.calibration_mode == "width" else "高度"
                print(f"[{mode_name}方向] 参考点 {len(ref_points)}: ({x}, {y})")
                
                # 在图像上绘制点
                cv.circle(self.current_frame, (x, y), 8, color, -1)
                cv.circle(self.current_frame, (x, y), 12, color, 2)
                
                # 如果选择了2个点，绘制参考线
                if len(ref_points) == 2:
                    pt1, pt2 = ref_points
                    cv.line(self.current_frame, pt1, pt2, color, 3)
                    
                    # 计算像素距离
                    pixel_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                    
                    # 在线上显示距离
                    mid_x = (pt1[0] + pt2[0]) // 2
                    mid_y = (pt1[1] + pt2[1]) // 2
                    cv.putText(self.current_frame, f"{pixel_length:.1f} px", 
                              (mid_x, mid_y - 20), cv.FONT_HERSHEY_SIMPLEX, 
                              0.8, color, 2)
                    
                    mode_name = "宽度" if self.calibration_mode == "width" else "高度"
                    print(f"\n[{mode_name}方向] 参考线像素长度: {pixel_length:.2f} 像素")
                    print(f"请输入[{mode_name}方向]参考物体的实际长度（米）:")
                
                cv.imshow("校准工具 - 选择参考物体", self.current_frame)
        
        elif event == cv.EVENT_MOUSEMOVE:
            # 实时显示鼠标位置
            temp_frame = self.current_frame.copy()
            
            # 绘制宽度方向的参考点（蓝色）
            for i, pt in enumerate(self.width_reference_points):
                cv.circle(temp_frame, pt, 8, (255, 0, 0), -1)
                cv.putText(temp_frame, f"W{i+1}", (pt[0] + 15, pt[1] - 15),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if len(self.width_reference_points) == 2:
                cv.line(temp_frame, self.width_reference_points[0], 
                       self.width_reference_points[1], (255, 0, 0), 2)
            
            # 绘制高度方向的参考点（红色）
            for i, pt in enumerate(self.height_reference_points):
                cv.circle(temp_frame, pt, 8, (0, 0, 255), -1)
                cv.putText(temp_frame, f"H{i+1}", (pt[0] + 15, pt[1] - 15),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if len(self.height_reference_points) == 2:
                cv.line(temp_frame, self.height_reference_points[0], 
                       self.height_reference_points[1], (0, 0, 255), 2)
            
            # 预览线
            if self.calibration_mode == "width" and len(self.width_reference_points) == 1:
                pt1 = self.width_reference_points[0]
                cv.line(temp_frame, pt1, (x, y), (255, 0, 0), 2)
            elif self.calibration_mode == "height" and len(self.height_reference_points) == 1:
                pt1 = self.height_reference_points[0]
                cv.line(temp_frame, pt1, (x, y), (0, 0, 255), 2)
            
            cv.circle(temp_frame, (x, y), 5, (255, 255, 0), 1)
            cv.putText(temp_frame, f"({x}, {y})", (x + 10, y - 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv.imshow("校准工具 - 选择参考物体", temp_frame)
    
    def load_frame(self, frame_num=None):
        """加载指定帧"""
        if frame_num is not None:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
            self.frame_number = frame_num
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            self.frame_number = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))
            
            # 绘制已标注的 IMAGE_POINTS 区域
            if self.image_points:
                pts = np.array(self.image_points, np.int32)
                cv.polylines(self.current_frame, [pts], True, (0, 255, 0), 2)
                # 填充半透明区域
                overlay = self.current_frame.copy()
                cv.fillPoly(overlay, [pts], (0, 255, 0))
                cv.addWeighted(overlay, 0.2, self.current_frame, 0.8, 0, self.current_frame)
                
                # 标注点编号
                labels = ["左上", "右上", "右下", "左下"]
                for i, (x, y) in enumerate(self.image_points):
                    cv.circle(self.current_frame, (x, y), 5, (0, 255, 0), -1)
                    cv.putText(self.current_frame, labels[i], (x + 10, y - 10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return True
        return False
    
    def calculate_scale(self, reference_length_meters, mode):
        """计算像素到米的转换比例"""
        if mode == "width":
            ref_points = self.width_reference_points
            if len(ref_points) != 2:
                return False
            pt1, pt2 = ref_points
            pixel_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            if pixel_length == 0:
                print("错误：参考点距离为0")
                return False
            self.width_scale = reference_length_meters / pixel_length
            self.width_length_meters = reference_length_meters
            print(f"\n[宽度方向] 校准结果:")
            print(f"  参考物体像素长度: {pixel_length:.2f} 像素")
            print(f"  参考物体实际长度: {reference_length_meters:.2f} 米")
            print(f"  转换比例: {self.width_scale:.6f} 米/像素")
            return True
        else:  # height
            ref_points = self.height_reference_points
            if len(ref_points) != 2:
                return False
            pt1, pt2 = ref_points
            pixel_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            if pixel_length == 0:
                print("错误：参考点距离为0")
                return False
            self.height_scale = reference_length_meters / pixel_length
            self.height_length_meters = reference_length_meters
            print(f"\n[高度方向] 校准结果:")
            print(f"  参考物体像素长度: {pixel_length:.2f} 像素")
            print(f"  参考物体实际长度: {reference_length_meters:.2f} 米")
            print(f"  转换比例: {self.height_scale:.6f} 米/像素")
            return True
    
    def calculate_world_points(self):
        """根据 IMAGE_POINTS 和比例计算 WORLD_POINTS（使用不同的横竖比例）"""
        if self.width_scale is None or self.height_scale is None:
            print("错误：请先完成宽度和高度方向的校准")
            return None
        
        # 计算 IMAGE_POINTS 区域的宽度和高度（像素）
        # 使用左上和右上计算宽度
        width_px = np.sqrt((self.image_points[1][0] - self.image_points[0][0])**2 + 
                          (self.image_points[1][1] - self.image_points[0][1])**2)
        
        # 使用左上和左下计算高度
        height_px = np.sqrt((self.image_points[3][0] - self.image_points[0][0])**2 + 
                           (self.image_points[3][1] - self.image_points[0][1])**2)
        
        # 使用不同的转换比例转换为米
        width_m = width_px * self.width_scale
        height_m = height_px * self.height_scale
        
        print(f"\n区域尺寸（使用不同横竖比例）:")
        print(f"  宽度: {width_px:.2f} 像素 × {self.width_scale:.6f} 米/像素 = {width_m:.2f} 米")
        print(f"  高度: {height_px:.2f} 像素 × {self.height_scale:.6f} 米/像素 = {height_m:.2f} 米")
        print(f"\n转换比例对比:")
        print(f"  宽度比例: {self.width_scale:.6f} 米/像素")
        print(f"  高度比例: {self.height_scale:.6f} 米/像素")
        print(f"  比例差异: {abs(self.width_scale - self.height_scale):.6f} 米/像素")
        
        # 生成 WORLD_POINTS（以左上角为原点）
        world_points = [
            (0, 0),
            (width_m, 0),
            (width_m, height_m),
            (0, height_m)
        ]
        
        return world_points
    
    def reset_reference(self):
        """重置参考点"""
        if self.calibration_mode == "width":
            self.width_reference_points = []
            self.width_scale = None
            print("\n已重置宽度方向参考点")
        else:
            self.height_reference_points = []
            self.height_scale = None
            print("\n已重置高度方向参考点")
        self.load_frame(self.frame_number)
    
    def run(self):
        """运行校准工具"""
        # 加载第一帧
        if not self.load_frame(0):
            print("错误：无法读取视频")
            return
        
        print("="*60)
        print("WORLD_POINTS 校准工具（支持不同横竖比例）")
        print("="*60)
        print(f"视频: {self.video_path}")
        print(f"已标注的 IMAGE_POINTS: {self.image_points}")
        print("\n操作说明:")
        print("  1. 首先校准宽度方向：选择水平方向的参考物体（如车道宽度）")
        print("  2. 然后校准高度方向：选择垂直方向的参考物体（如道路标线长度）")
        print("  3. 工具会使用不同的横竖比例自动计算 WORLD_POINTS")
        print("\n参考物体建议:")
        print("  宽度方向:")
        print("    - 标准车道宽度: 3.66 米（美国高速）")
        print("    - 标准车道宽度: 3.5 米（中国城市道路）")
        print("  高度方向:")
        print("    - 道路标线长度: 2-3 米")
        print("    - 标准车辆长度: 4.5 米（小轿车）")
        print("\n快捷键:")
        print("  - 鼠标左键: 选择参考点（需要2个点）")
        print("  - 按 'w': 切换到宽度方向校准")
        print("  - 按 'h': 切换到高度方向校准")
        print("  - 按 'r': 重置当前方向的参考点")
        print("  - 按 'n': 下一帧")
        print("  - 按 'p': 上一帧")
        print("  - 按 'q' 或 ESC: 退出")
        print("="*60)
        print(f"\n当前校准模式: {'宽度方向' if self.calibration_mode == 'width' else '高度方向'}")
        print("提示: 蓝色(W)表示宽度方向，红色(H)表示高度方向")
        
        window_name = "校准工具 - 选择参考物体"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            # 更新窗口标题显示当前模式
            mode_name = "宽度方向" if self.calibration_mode == "width" else "高度方向"
            status = f"校准工具 - {mode_name}"
            if self.width_scale is not None:
                status += " [宽度✓]"
            if self.height_scale is not None:
                status += " [高度✓]"
            cv.setWindowTitle(window_name, status)
            cv.imshow(window_name, self.current_frame)
            
            key = cv.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' 或 ESC
                break
            elif key == ord('w'):  # 切换到宽度方向
                self.calibration_mode = "width"
                print("\n已切换到宽度方向校准模式")
            elif key == ord('h'):  # 切换到高度方向
                self.calibration_mode = "height"
                print("\n已切换到高度方向校准模式")
            elif key == ord('r'):  # 重置
                self.reset_reference()
            elif key == ord('n'):  # 下一帧
                if self.frame_number < self.total_frames - 1:
                    self.frame_number += 1
                    self.load_frame(self.frame_number)
                    print(f"\n已切换到第 {self.frame_number} 帧")
            elif key == ord('p'):  # 上一帧
                if self.frame_number > 0:
                    self.frame_number -= 1
                    self.load_frame(self.frame_number)
                    print(f"\n已切换到第 {self.frame_number} 帧")
            elif (self.calibration_mode == "width" and len(self.width_reference_points) == 2 and self.width_scale is None) or \
                 (self.calibration_mode == "height" and len(self.height_reference_points) == 2 and self.height_scale is None):
                # 当选择了2个参考点后，提示输入实际长度
                try:
                    mode_name = "宽度" if self.calibration_mode == "width" else "高度"
                    length = float(input(f"请输入[{mode_name}方向]参考物体的实际长度（米）: "))
                    if length > 0:
                        if self.calculate_scale(length, self.calibration_mode):
                            # 如果两个方向都校准完成，计算 WORLD_POINTS
                            if self.width_scale is not None and self.height_scale is not None:
                                world_points = self.calculate_world_points()
                                if world_points:
                                    self.save_results(world_points)
                            else:
                                remaining = "高度" if self.calibration_mode == "width" else "宽度"
                                print(f"\n请继续校准{remaining}方向")
                    else:
                        print("长度必须大于0")
                except ValueError:
                    print("请输入有效的数字")
                except KeyboardInterrupt:
                    break
        
        self.cap.release()
        cv.destroyAllWindows()
    
    def save_results(self, world_points):
        """保存校准结果"""
        print("\n" + "="*60)
        print("校准完成！")
        print("="*60)
        print("\nIMAGE_POINTS:")
        print(f"  {self.image_points}")
        print("\nWORLD_POINTS:")
        print(f"  {world_points}")
        print("\n复制以下代码到 utils/constants.py:")
        print("-"*60)
        print(f"IMAGE_POINTS = {self.image_points}")
        print(f"WORLD_POINTS = {world_points}")
        print("-"*60)
        
        # 保存到文件
        output_file = "calibrated_world_points.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 校准的 WORLD_POINTS（使用不同横竖比例）\n")
            f.write(f"# 视频: {self.video_path}\n")
            f.write(f"# 宽度方向参考物体长度: {self.width_length_meters} 米\n")
            f.write(f"# 宽度方向转换比例: {self.width_scale:.6f} 米/像素\n")
            f.write(f"# 高度方向参考物体长度: {self.height_length_meters} 米\n")
            f.write(f"# 高度方向转换比例: {self.height_scale:.6f} 米/像素\n")
            f.write(f"\nIMAGE_POINTS = {self.image_points}\n")
            f.write(f"WORLD_POINTS = {world_points}\n")
        
        print(f"\n✓ 已保存到文件: {output_file}")
        print("="*60)


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='WORLD_POINTS 校准工具')
    parser.add_argument('--video', '-v', type=str, default=None,
                        help='视频文件路径（可选）')
    parser.add_argument('--points', '-p', type=str, default=None,
                        help='IMAGE_POINTS（可选，格式: "[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]"）')
    args = parser.parse_args()
    
    # 获取视频路径
    if args.video:
        video_path = args.video
        if not os.path.isabs(video_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_path = os.path.join(base_dir, video_path)
    else:
        try:
            video_path = download_video_if_needed()
        except Exception as e:
            print(f"错误：无法获取视频路径: {e}")
            return
    
    # 获取 IMAGE_POINTS
    if args.points:
        try:
            image_points = eval(args.points)
        except:
            print("错误：IMAGE_POINTS 格式不正确")
            return
    else:
        # 从 constants.py 读取
        image_points = IMAGE_POINTS
    
    if not image_points or len(image_points) != 4:
        print("错误：需要4个 IMAGE_POINTS")
        print("请先运行 annotate.py 标注点，或使用 --points 参数")
        return
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return
    
    # 创建校准工具并运行
    calibrator = WorldPointsCalibrator(video_path, image_points)
    calibrator.run()


if __name__ == "__main__":
    main()
