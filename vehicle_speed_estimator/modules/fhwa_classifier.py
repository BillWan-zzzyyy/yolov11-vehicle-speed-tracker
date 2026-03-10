"""
FHWA 13-Class 车辆分类映射模块

将 YOLO/COCO 检测类别映射到 FHWA (Federal Highway Administration) 13 类标准体系。
FHWA 分类基于车轴数量和轮胎配置，详见:
https://www.fhwa.dot.gov/policyinformation/vehclass.cfm
"""

FHWA_CLASSES = {
    1:  {"name": "Motorcycles",               "description": "2-3 wheeled motorized vehicles"},
    2:  {"name": "Passenger Cars",             "description": "Sedans, coupes, station wagons"},
    3:  {"name": "Other 2-Axle 4-Tire",        "description": "Pickups, SUVs, vans"},
    4:  {"name": "Buses",                      "description": "Transit, school, intercity buses"},
    5:  {"name": "2-Axle 6-Tire Single Unit",   "description": "Larger delivery trucks, RVs"},
    6:  {"name": "3-Axle Single Unit",          "description": "Medium-duty trucks with 3 axles"},
    7:  {"name": "4+ Axle Single Unit",         "description": "Large single-unit trucks"},
    8:  {"name": "3-4 Axle Single-Trailer",     "description": "Smaller semi-trucks"},
    9:  {"name": "5-Axle Single-Trailer",       "description": "Standard 18-wheelers"},
    10: {"name": "6+ Axle Single-Trailer",      "description": "Heavy single-trailer trucks"},
    11: {"name": "5- Axle Multi-Trailer",       "description": "Smaller multi-trailer trucks"},
    12: {"name": "6-Axle Multi-Trailer",        "description": "Medium multi-trailer trucks"},
    13: {"name": "7+ Axle Multi-Trailer",       "description": "Heavy multi-trailer trucks"},
}

COCO_TO_FHWA_MAP = {
    "motorcycle": {"fhwa_class": 1, "fhwa_name": "Motorcycles"},
    "car":        {"fhwa_class": 2, "fhwa_name": "Passenger Cars"},
    "bus":        {"fhwa_class": 4, "fhwa_name": "Buses"},
    "truck":      {"fhwa_class": 5, "fhwa_name": "2-Axle 6-Tire Single Unit"},
}

_UNKNOWN_FHWA = {"fhwa_class": None, "fhwa_name": "Unknown"}


def coco_to_fhwa(coco_class_name: str) -> dict:
    """
    将 COCO 类别名称映射到 FHWA 分类。

    Args:
        coco_class_name: COCO 检测类别名称，如 "car", "truck", "bus", "motorcycle"

    Returns:
        包含 fhwa_class (int|None) 和 fhwa_name (str) 的字典

    Example:
        coco_to_fhwa("car")   # {"fhwa_class": 2, "fhwa_name": "Passenger Cars"}
        coco_to_fhwa("truck") # {"fhwa_class": 5, "fhwa_name": "2-Axle 6-Tire Single Unit"}
    """
    return COCO_TO_FHWA_MAP.get(coco_class_name, _UNKNOWN_FHWA)


def format_fhwa_label(fhwa_info: dict) -> str:
    """
    格式化 FHWA 分类信息为可视化标签字符串。

    Args:
        fhwa_info: coco_to_fhwa() 返回的字典

    Returns:
        格式化标签，如 "CL-2: Passenger Cars"；未知类别返回 "Unknown"

    Example:
        format_fhwa_label({"fhwa_class": 2, "fhwa_name": "Passenger Cars"})
        # "CL-2: Passenger Cars"
    """
    fhwa_class = fhwa_info.get("fhwa_class")
    fhwa_name = fhwa_info.get("fhwa_name", "Unknown")
    if fhwa_class is not None:
        return f"CL-{fhwa_class}: {fhwa_name}"
    return "Unknown"
