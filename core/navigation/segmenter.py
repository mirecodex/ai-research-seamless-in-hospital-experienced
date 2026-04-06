from typing import Optional

from pydantic import BaseModel

from .graph import HospitalGraph
from app.utils.GeoUtils import cross_product, angle_between, classify_turn


class RouteSegment(BaseModel):
    nodes: list[str]
    floor: int
    distance: float
    direction: str = "straight"
    landmarks: list[str] = []
    start_node: str
    end_node: str
    floor_change: Optional[dict] = None


class RouteSegmenter:
    TURN_THRESHOLD_DEG = 30.0
    MAX_SEGMENT_METERS = 80.0
    MAP_RATIO = 20.0

    def segment(
        self,
        graph: HospitalGraph,
        path: list[str],
        profile: str = "default",
    ) -> list[RouteSegment]:
        if len(path) < 2:
            return []

        floor_groups, transitions = self._split_by_floor(graph, path)
        segments: list[RouteSegment] = []

        for gi, group in enumerate(floor_groups):
            turn_indices = self._detect_turns(graph, group)
            sub_segments = self._split_at_indices(graph, group, turn_indices)

            # Attach floor_change to the first segment of a group that resulted from a floor transition
            if sub_segments and gi in transitions:
                sub_segments[0].floor_change = transitions[gi]

            for seg in sub_segments:
                split = self._split_by_distance(seg)
                segments.extend(split)

        return segments

    def _detect_turns(self, graph: HospitalGraph, path: list[str]) -> list[int]:
        """Return indices in path where a significant turn occurs."""
        turn_indices: list[int] = []

        for i in range(1, len(path) - 1):
            a = graph.nodes[path[i - 1]]
            b = graph.nodes[path[i]]
            c = graph.nodes[path[i + 1]]

            angle = angle_between((a.x, a.y), (b.x, b.y), (c.x, c.y))
            if abs(angle) >= self.TURN_THRESHOLD_DEG:
                turn_indices.append(i)

        return turn_indices

    def _split_by_floor(
        self, graph: HospitalGraph, path: list[str]
    ) -> tuple[list[list[str]], dict[int, dict]]:
        """Split path into per-floor groups and record floor transitions.

        Returns (groups, transitions) where transitions maps group index to
        {"from_floor": int, "to_floor": int, "via": "elevator"|"stairs"|"corridor"}.
        """
        if not path:
            return [], {}

        groups: list[list[str]] = [[path[0]]]
        transitions: dict[int, dict] = {}

        for i in range(1, len(path)):
            prev_node = graph.nodes[path[i - 1]]
            curr_node = graph.nodes[path[i]]
            if curr_node.floor != prev_node.floor:
                via = "corridor"
                if curr_node.category == "ELEVATOR" or prev_node.category == "ELEVATOR":
                    via = "elevator"
                elif curr_node.category == "STAIRS" or prev_node.category == "STAIRS":
                    via = "stairs"
                elif curr_node.type == "elevator" or prev_node.type == "elevator":
                    via = "elevator"
                elif curr_node.type == "stairs" or prev_node.type == "stairs":
                    via = "stairs"

                group_idx = len(groups)
                transitions[group_idx] = {
                    "from_floor": prev_node.floor,
                    "to_floor": curr_node.floor,
                    "via": via,
                }
                groups.append([path[i]])
            else:
                groups[-1].append(path[i])

        return groups, transitions

    def _split_at_indices(
        self,
        graph: HospitalGraph,
        path: list[str],
        turn_indices: list[int],
    ) -> list[RouteSegment]:
        if not path:
            return []

        split_points = sorted(set(turn_indices))
        segments: list[RouteSegment] = []
        start = 0

        for idx in split_points:
            if idx <= start:
                continue
            chunk = path[start : idx + 1]
            segments.append(self._build_segment(graph, chunk))
            start = idx

        if start < len(path) - 1:
            chunk = path[start:]
            segments.append(self._build_segment(graph, chunk))

        if not segments and len(path) >= 2:
            segments.append(self._build_segment(graph, path))

        return segments

    def _split_by_distance(self, segment: RouteSegment) -> list[RouteSegment]:
        max_px = self.MAX_SEGMENT_METERS * self.MAP_RATIO
        if segment.distance <= max_px:
            return [segment]
        # For long segments, split in half recursively
        mid = len(segment.nodes) // 2
        if mid < 1:
            return [segment]
        first = RouteSegment(
            nodes=segment.nodes[: mid + 1],
            floor=segment.floor,
            distance=segment.distance / 2,
            direction=segment.direction,
            landmarks=[],
            start_node=segment.nodes[0],
            end_node=segment.nodes[mid],
            floor_change=segment.floor_change,
        )
        second = RouteSegment(
            nodes=segment.nodes[mid:],
            floor=segment.floor,
            distance=segment.distance / 2,
            direction="straight",
            landmarks=[],
            start_node=segment.nodes[mid],
            end_node=segment.nodes[-1],
        )
        return self._split_by_distance(first) + self._split_by_distance(second)

    def _build_segment(
        self, graph: HospitalGraph, nodes: list[str]
    ) -> RouteSegment:
        total_dist = 0.0
        for i in range(len(nodes) - 1):
            total_dist += graph.euclidean_distance(nodes[i], nodes[i + 1])

        direction = "straight"
        if len(nodes) >= 3:
            a = graph.nodes[nodes[-3]]
            b = graph.nodes[nodes[-2]]
            c = graph.nodes[nodes[-1]]
            angle = angle_between((a.x, a.y), (b.x, b.y), (c.x, c.y))
            direction = classify_turn(angle)

        landmarks = []
        for nid in nodes:
            node = graph.nodes[nid]
            if node.type != "junction" and node.name:
                landmarks.append(node.name)

        return RouteSegment(
            nodes=nodes,
            floor=graph.nodes[nodes[0]].floor,
            distance=round(total_dist, 1),
            direction=direction,
            landmarks=landmarks,
            start_node=nodes[0],
            end_node=nodes[-1],
        )
