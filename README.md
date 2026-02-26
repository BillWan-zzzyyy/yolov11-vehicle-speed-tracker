# yolov11-vehicle-speed-state-tracker for USDOT ARPA-I Ideas Challenge
## Introduction
In this repository, a yolov11-based vehicle speed and state tracker is established. This is a computer vision pipeline that performs real-time vehicle detection, tracking, and speed estimation from video footage. Leveraging YOLOv11 for object detection and ByteTrack for multi-object tracking, it maps vehicle positions from camera space to world space using perspective transformation, enabling accurate speed calculation in mph, the speeding detection, and abnormal state detection.

Designed for use in road surveillance, intelligent traffic systems, and smart city infrastructure, this system integrates with the Supervision library for real-time annotation and visualization.

This repository is built on the foundation of [**YOLOv11-vehicle-speed-tracker.**](https://github.com/krishnapriya-nynaru/yolov11-vehicle-speed-tracker)


## Table of Contents
- [Features](#features)
- [Project Structure](#ProjectStructure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Features
- 📦 Object detection with YOLOv11
- 🧭 Perspective transformation for accurate distance mapping
- 🏎️ Speed estimation in real-world units (mph)
- 🔁 Multi-object tracking using ByteTrack
- 🖥️ Real-time annotation with bounding boxes, traces, and speed labels
- 🛞 Speeding and wrong lane detection
- 🛠️ Modular and production-ready Python package structure

## Project Structure
```bash
vehicle_speed_estimation/
├── annotate.py                  # Annotation entry script
├── annotated_lane_lines.txt     # Lane line coordinates
├── annotated_points.txt         # Image point coordinates
├── ANNOTATION_GUIDE.md          # Annotation instructions
├── assets                       # Media input assets
├── calibrate.py                 # Calibration entry script
├── calibrated_world_points.txt  # World calibration points
├── config                       # Configuration package
│   ├── __init__.py              
│   └── settings.py              # Runtime settings
├── main.py                      # main code
├── models                       # Model checkpoints
│   ├── yolo11n.pt               
│   └── yolo11s.pt               
├── modules                      # Core logic modules
│   ├── __init__.py              
│   ├── annotators.py            # Drawing and labels
│   ├── data_recorder.py         # Detection data logging
│   ├── lane_assigner.py         # Lane assignment logic
│   ├── mapping.py               # Perspective mapping logic
│   ├── speedometer.py           # Speed estimation logic
│   └── video_recorder.py        # Output video writer
├── results                      # Run outputs
├── utils                        
│   ├── __init__.py              
│   ├── annotate_points.py       # Point annotation helpers
│   ├── bytetracker.yaml         # ByteTrack config
│   ├── calibrate_world_points.py # World-point calibration helper
│   ├── constants.py             # Shared constants
│   └── downloader.py            # Input video downloader
└── zone                         # Zone logic package
    ├── __init__.py              
    └── zone_trigger.py          
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/BillWan-zzzyyy/yolov11-vehicle-speed-tracker.git
2. Install required packages and activate environment
    ```bash
    conda env create -f environment.yml
    conda activate yolov11-vehicle-speed-tracker
3. Change to Project Directory:
    ```bash
    cd vehicle_speed_estimator

## Usage
Run the script with Python
```bash
python main.py
```
`Note:`
You can change the YOLOv11 model variant in config/settings.py to experiment with detection performance vs. speed.

For Example:
```bash
# config/settings.py
MODEL_PATH = "yolo11s.pt"  # Or "yolo11m.pt", "yolo11n.pt"
```
- yolo11n.pt – Fastest, least accurate
- yolo11s.pt – Good balance
- yolo11m.pt – More accurate, slower

Try each and observe the trade-off between FPS and accuracy in your environment.

### Improvement
如果需要更改其他视频，需要先进行比例转换从而计算速度
```bash
# 启动对image_point（图片点 跟踪区域）的标注
python annotate.py
# 按顺序点击4个点：1. 左上角 2. 右上角 3. 右下角 4. 左下角 s保存 q退出
# 复制image_point到constant.py中
# 如果有需要，需要按照指示重新标注车道线
```

启动world_point标注 （比例转换）
```bash
python calibrate.py
# 按 w 切换到宽度模式,选择水平方向的参考物体（如车道宽度; 按 h 切换到高度模式.选择垂直方向的参考物体（如道路标线间隔9.14）
# 将 WORLD_POINTS复制到constant.py中
```

## Results



https://github.com/user-attachments/assets/e6aad29b-c3f7-411c-909a-fac3fc3e95d0




## Contributing 
Contributions are welcome! To contribute to this project:
1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and ensure the code passes all tests.
4. Submit a pull request with a detailed description of your changes.

If you have any suggestions for improvements or features, feel free to open an issue!

## Acknowledgments
- [**YOLOv11-vehicle-speed-tracker.**](https://github.com/krishnapriya-nynaru/yolov11-vehicle-speed-tracker)
- [**YOLOv11 for object detection.**](https://github.com/ultralytics/yolov11)
- [**ByteTracker for Multi object tracking.**](https://github.com/FoundationVision/ByteTrack)
- [**OpenCV for computer vision functionalities.**](https://opencv.org/)
- [**Supervision by Roboflow, for seamless computer vision annotation tools (bounding boxes, traces, and on-frame labels).**](https://github.com/roboflow/supervision)

### 🎉 Happy coding!
