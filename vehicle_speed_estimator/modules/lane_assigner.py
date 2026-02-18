"""
车道判定模块 - 支持 x 阈值和线段分界两种车道判定方式。
"""

from typing import Iterable, List, Optional, Sequence, Tuple


class LaneAssigner:
    """
    根据车道分界线判定车道编号。

    分界线需按从左到右递增输入，若有 k 条分界线，则车道数为 k+1。
    车道编号从 1 开始。

    Example:
        assigner = LaneAssigner([900, 1300])
        lane_1 = assigner.get_lane_id(820)   # 1
        lane_2 = assigner.get_lane_id(980)   # 2
        lane_3 = assigner.get_lane_id(1500)  # 3
    """

    def __init__(
        self,
        lane_boundaries_x: Optional[Iterable[float]] = None,
        lane_boundary_lines: Optional[Iterable[Sequence[Sequence[float]]]] = None,
        mode: str = "x_threshold",
        emergency_lane_enabled: bool = False,
        emergency_lane_between_boundaries: Optional[Sequence[int]] = None,
        emergency_lane_rules: Optional[Sequence[dict]] = None
    ):
        """
        初始化车道判定器。

        Args:
            lane_boundaries_x: 车道分界线 x 坐标序列（从左到右递增）。
            lane_boundary_lines: 车道分界线段列表，每条线段为 [(x1,y1),(x2,y2)]。
            mode: 判定模式，支持 "x_threshold" 或 "line_segments"。
            emergency_lane_enabled: 是否启用应急车道判定。
            emergency_lane_between_boundaries: 应急车道位于哪两条边界线之间，索引从1开始。
            emergency_lane_rules: 新版应急车道规则列表。非空时优先于 legacy 配置。
        """
        boundaries = list(lane_boundaries_x) if lane_boundaries_x is not None else []
        self.boundaries: List[float] = self._validate_and_sort_boundaries(boundaries)
        self.mode = mode if mode in {"x_threshold", "line_segments"} else "x_threshold"
        self.lane_boundary_lines = self._validate_lane_lines(lane_boundary_lines)
        self.emergency_lane_enabled = bool(emergency_lane_enabled)
        self.emergency_lane_between_boundaries = self._validate_emergency_boundaries(
            emergency_lane_between_boundaries
        )
        self.emergency_lane_rules = self._parse_emergency_lane_rules(emergency_lane_rules)

    def _validate_and_sort_boundaries(self, boundaries: List[float]) -> List[float]:
        """
        校验并标准化分界线配置。

        Args:
            boundaries: 原始分界线列表。

        Returns:
            清洗后的升序分界线列表（float）。
        """
        cleaned: List[float] = []
        for value in boundaries:
            if value is None:
                continue
            cleaned.append(float(value))

        # 自动排序，避免因配置顺序错误导致车道编号异常。
        cleaned.sort()
        return cleaned

    def _validate_lane_lines(self, lane_boundary_lines):
        """
        校验并标准化线段车道边界。

        Args:
            lane_boundary_lines: 原始线段列表。

        Returns:
            标准化后的线段列表，每项为 ((x1,y1),(x2,y2))。
        """
        validated = []
        if lane_boundary_lines is None:
            return validated

        for line in lane_boundary_lines:
            if line is None or len(line) != 2:
                continue
            p1, p2 = line[0], line[1]
            if p1 is None or p2 is None or len(p1) != 2 or len(p2) != 2:
                continue
            try:
                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])
            except (TypeError, ValueError):
                continue
            validated.append(((x1, y1), (x2, y2)))
        return validated

    def _validate_emergency_boundaries(self, emergency_lane_between_boundaries):
        """
        校验应急车道边界索引配置。

        Args:
            emergency_lane_between_boundaries: 例如 (1, 2)。

        Returns:
            合法配置返回 (left_idx, right_idx)，否则返回 None。
        """
        if emergency_lane_between_boundaries is None:
            return None
        if len(emergency_lane_between_boundaries) != 2:
            return None

        try:
            left_idx = int(emergency_lane_between_boundaries[0])
            right_idx = int(emergency_lane_between_boundaries[1])
        except (TypeError, ValueError):
            return None

        if left_idx < 1 or right_idx < 1 or left_idx >= right_idx:
            return None
        return left_idx, right_idx

    def _parse_emergency_lane_rules(self, emergency_lane_rules):
        """
        解析新版应急车道规则。

        Args:
            emergency_lane_rules: 规则列表。

        Returns:
            标准化后的规则列表。非法规则会被忽略。
        """
        parsed_rules = []
        if emergency_lane_rules is None:
            return parsed_rules

        for rule in emergency_lane_rules:
            if not isinstance(rule, dict):
                continue
            rule_type = str(rule.get("type", "")).strip().lower()
            if rule_type == "between":
                try:
                    left = int(rule.get("left"))
                    right = int(rule.get("right"))
                except (TypeError, ValueError):
                    continue
                parsed_rules.append({"type": "between", "left": left, "right": right})
            elif rule_type == "left_of":
                try:
                    boundary = int(rule.get("boundary"))
                except (TypeError, ValueError):
                    continue
                parsed_rules.append({"type": "left_of", "boundary": boundary})
            elif rule_type == "right_of":
                try:
                    boundary = int(rule.get("boundary"))
                except (TypeError, ValueError):
                    continue
                parsed_rules.append({"type": "right_of", "boundary": boundary})
        return parsed_rules

    def _get_boundary_count(self) -> int:
        """获取当前判定模式下的边界数量。"""
        if self.mode == "line_segments" and self.lane_boundary_lines:
            return len(self.lane_boundary_lines)
        return len(self.boundaries)

    def _get_emergency_lane_ids(self):
        """
        计算应急车道对应的 lane_id 集合。

        Returns:
            应急车道编号集合，配置非法或未启用时返回空集合。
        """
        if not self.emergency_lane_enabled or self.emergency_lane_between_boundaries is None:
            if not self.emergency_lane_enabled:
                return set()

        boundary_count = self._get_boundary_count()
        if boundary_count <= 0:
            return set()

        # 新规则优先：支持 left_of / between / right_of 三种表达。
        if self.emergency_lane_rules:
            lane_ids = set()
            max_lane_id = boundary_count + 1
            for rule in self.emergency_lane_rules:
                rule_type = rule["type"]
                if rule_type == "left_of":
                    boundary = rule["boundary"]
                    if 1 <= boundary <= boundary_count:
                        lane_ids.update(range(1, boundary + 1))
                elif rule_type == "right_of":
                    boundary = rule["boundary"]
                    if 1 <= boundary <= boundary_count:
                        lane_ids.update(range(boundary + 1, max_lane_id + 1))
                elif rule_type == "between":
                    left_idx = rule["left"]
                    right_idx = rule["right"]
                    if 1 <= left_idx < right_idx <= boundary_count:
                        # 边界线 i 与 i+1 之间是 lane i+1。
                        lane_ids.update(range(left_idx + 1, right_idx + 1))
            return lane_ids

        # legacy 回退：仅支持 between(i, j)
        if self.emergency_lane_between_boundaries is None:
            return set()

        left_idx, right_idx = self.emergency_lane_between_boundaries
        if right_idx > boundary_count:
            return set()
        return set(range(left_idx + 1, right_idx + 1))

    def _project_line_x_at_y(self, line: Tuple[Tuple[float, float], Tuple[float, float]], y_value: float) -> Optional[float]:
        """
        计算给定 y 处的边界线 x 值。超出线段 y 范围时使用端点夹紧。

        Args:
            line: 单条边界线段。
            y_value: 目标 y 坐标。

        Returns:
            对应 x 值；若输入异常返回 None。
        """
        (x1, y1), (x2, y2) = line
        try:
            y = float(y_value)
        except (TypeError, ValueError):
            return None

        # 水平线特殊处理，取中点 x 作为边界近似。
        if y1 == y2:
            return (x1 + x2) / 2.0

        # 限制到线段 y 范围，避免外推导致边界漂移。
        y_min, y_max = min(y1, y2), max(y1, y2)
        clamped_y = min(max(y, y_min), y_max)
        t = (clamped_y - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)

    def get_lane_id(self, x_position: Optional[float]) -> Optional[int]:
        """
        根据 x 坐标计算车道编号。

        Args:
            x_position: 车辆当前位置的图像 x 坐标。

        Returns:
            车道编号（从 1 开始）；若输入无效则返回 None。

        Example:
            assigner = LaneAssigner([100.0, 200.0])
            assigner.get_lane_id(50.0)    # 1
            assigner.get_lane_id(150.0)   # 2
            assigner.get_lane_id(260.0)   # 3
        """
        if x_position is None:
            return None

        try:
            x_value = float(x_position)
        except (TypeError, ValueError):
            return None

        for idx, boundary in enumerate(self.boundaries):
            if x_value < boundary:
                return idx + 1
        return len(self.boundaries) + 1

    def get_lane_id_from_point(self, point: Optional[Sequence[float]]) -> Optional[int]:
        """
        根据二维点坐标计算车道编号。

        Args:
            point: 图像坐标点，格式为 (x, y)。

        Returns:
            车道编号（从 1 开始）；输入无效返回 None。

        Example:
            assigner = LaneAssigner(
                lane_boundary_lines=[[(900, 200), (1000, 900)]],
                mode="line_segments"
            )
            lane_id = assigner.get_lane_id_from_point((850, 500))  # 1
        """
        if point is None or len(point) < 2:
            return None

        try:
            x_value = float(point[0])
            y_value = float(point[1])
        except (TypeError, ValueError):
            return None

        if self.mode == "line_segments" and self.lane_boundary_lines:
            projected_boundaries = []
            for line in self.lane_boundary_lines:
                projected_x = self._project_line_x_at_y(line, y_value)
                if projected_x is not None:
                    projected_boundaries.append(projected_x)

            if projected_boundaries:
                projected_boundaries.sort()
                for idx, boundary_x in enumerate(projected_boundaries):
                    if x_value < boundary_x:
                        return idx + 1
                return len(projected_boundaries) + 1

        # 线段模式不可用时，自动回退到 x 阈值模式。
        return self.get_lane_id(x_value)

    def is_emergency_lane(self, lane_id: Optional[int]) -> bool:
        """
        判断车道编号是否属于应急车道。

        Args:
            lane_id: 车道编号（从1开始）。

        Returns:
            属于应急车道返回 True，否则 False。
        """
        if lane_id is None:
            return False

        emergency_lane_ids = self._get_emergency_lane_ids()
        return lane_id in emergency_lane_ids

