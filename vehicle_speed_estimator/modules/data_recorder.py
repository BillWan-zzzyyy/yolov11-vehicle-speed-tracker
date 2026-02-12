"""
数据记录模块 - 实时记录车辆ID、速度和轨迹坐标
"""
import json
import csv
import os
from datetime import datetime
from collections import defaultdict
import numpy as np


class VehicleDataRecorder:
    def __init__(self, output_dir="results"):
        """
        初始化数据记录器
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储车辆数据
        # 格式: {vehicle_id: [{frame, timestamp, speed, image_coords, world_coords}, ...]}
        self.vehicle_data = defaultdict(list)
        
        # 当前帧号
        self.current_frame = 0
        
        # 记录开始时间
        self.start_time = datetime.now()
        
    def record_vehicle(self, vehicle_id, speed, image_trace, world_trace=None):
        """
        记录车辆数据
        
        参数:
            vehicle_id: 车辆ID
            speed: 当前速度 (mile/h)
            image_trace: 图像坐标轨迹 [(x1, y1), (x2, y2), ...]
            world_trace: 世界坐标轨迹 [(x1, y1), (x2, y2), ...] (可选)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 获取当前位置（轨迹的最后一个点）
        # 检查 image_trace 是否为空（numpy 数组需要特殊处理）
        if image_trace is not None and len(image_trace) > 0:
            current_image_pos = tuple(image_trace[-1]) if isinstance(image_trace, np.ndarray) else image_trace[-1]
        else:
            current_image_pos = None
            
        # 检查 world_trace 是否为空
        if world_trace is not None and len(world_trace) > 0:
            current_world_pos = tuple(world_trace[-1]) if isinstance(world_trace, np.ndarray) else world_trace[-1]
        else:
            current_world_pos = None
        
        # 记录数据
        record = {
            "frame": self.current_frame,
            "timestamp": timestamp,
            "speed_mph": float(speed),
            "image_position": {
                "x": float(current_image_pos[0]) if current_image_pos else None,
                "y": float(current_image_pos[1]) if current_image_pos else None
            },
            "world_position": {
                "x": float(current_world_pos[0]) if current_world_pos else None,
                "y": float(current_world_pos[1]) if current_world_pos else None
            },
            "image_trace": [[float(x), float(y)] for x, y in image_trace] if (image_trace is not None and len(image_trace) > 0) else [],
            "world_trace": [[float(x), float(y)] for x, y in world_trace] if (world_trace is not None and len(world_trace) > 0) else []
        }
        
        self.vehicle_data[vehicle_id].append(record)
    
    def next_frame(self):
        """进入下一帧"""
        self.current_frame += 1
    
    def save_to_csv(self, filename=None):
        """
        保存数据到CSV文件
        
        参数:
            filename: 输出文件名（可选）
        """
        if filename is None:
            timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"vehicle_data_{timestamp_str}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow([
                "Vehicle_ID", "Frame", "Timestamp", "Speed_mph",
                "Image_X", "Image_Y", "World_X", "World_Y",
                "Image_Trace_Length", "World_Trace_Length"
            ])
            
            # 写入数据
            for vehicle_id, records in self.vehicle_data.items():
                for record in records:
                    writer.writerow([
                        vehicle_id,
                        record["frame"],
                        record["timestamp"],
                        record["speed_mph"],
                        record["image_position"]["x"],
                        record["image_position"]["y"],
                        record["world_position"]["x"],
                        record["world_position"]["y"],
                        len(record["image_trace"]),
                        len(record["world_trace"])
                    ])
        
        return filename
    
    def save_to_json(self, filename=None):
        """
        保存数据到JSON文件（包含完整轨迹信息）
        
        参数:
            filename: 输出文件名（可选）
        """
        if filename is None:
            timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"vehicle_data_{timestamp_str}.json")
        
        # 准备输出数据
        output_data = {
            "metadata": {
                "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_frames": self.current_frame,
                "total_vehicles": len(self.vehicle_data)
            },
            "vehicles": {}
        }
        
        # 转换数据格式
        for vehicle_id, records in self.vehicle_data.items():
            output_data["vehicles"][str(vehicle_id)] = {
                "total_records": len(records),
                "records": records
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def save_summary(self, filename=None):
        """
        保存车辆摘要信息（每辆车的基本统计）
        
        参数:
            filename: 输出文件名（可选）
        """
        if filename is None:
            timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"vehicle_summary_{timestamp_str}.csv")
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Vehicle_ID", "Total_Frames", "First_Frame", "Last_Frame",
                "Max_Speed_mph", "Min_Speed_mph", "Avg_Speed_mph",
                "Total_Distance_m", "Duration_seconds"
            ])
            
            for vehicle_id, records in self.vehicle_data.items():
                if not records:
                    continue
                
                speeds = [r["speed_mph"] for r in records if r["speed_mph"] > 0]
                world_positions = [r["world_position"] for r in records 
                                 if r["world_position"]["x"] is not None]
                
                # 计算总距离
                total_distance = 0.0
                if len(world_positions) > 1:
                    for i in range(1, len(world_positions)):
                        x1, y1 = world_positions[i-1]["x"], world_positions[i-1]["y"]
                        x2, y2 = world_positions[i]["x"], world_positions[i]["y"]
                        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            total_distance += distance
                
                # 计算持续时间（基于帧数）
                duration = (records[-1]["frame"] - records[0]["frame"]) / 30.0  # 假设30fps
                
                writer.writerow([
                    vehicle_id,
                    len(records),
                    records[0]["frame"],
                    records[-1]["frame"],
                    max(speeds) if speeds else 0,
                    min(speeds) if speeds else 0,
                    sum(speeds) / len(speeds) if speeds else 0,
                    total_distance,
                    duration
                ])
        
        return filename
    
    def save_all(self):
        """保存所有数据文件"""
        csv_file = self.save_to_csv()
        json_file = self.save_to_json()
        summary_file = self.save_summary()
        
        print(f"\n数据已保存:")
        print(f"  - CSV文件（简化数据）: {csv_file}")
        print(f"  - JSON文件（完整轨迹）: {json_file}")
        print(f"  - 摘要文件（统计信息）: {summary_file}")
        
        return csv_file, json_file, summary_file
    
    def get_statistics(self):
        """获取统计信息"""
        total_vehicles = len(self.vehicle_data)
        total_records = sum(len(records) for records in self.vehicle_data.values())
        
        return {
            "total_vehicles": total_vehicles,
            "total_records": total_records,
            "total_frames": self.current_frame,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        }
