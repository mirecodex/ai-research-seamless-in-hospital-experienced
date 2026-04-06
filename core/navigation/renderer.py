from .graph import HospitalGraph
from .segmenter import RouteSegment
from app.utils.GeoUtils import bounding_box, angle_between, classify_turn
from app.utils.SVGUtils import (
    inject_route_overlay,
    inject_arrows,
    inject_labels,
    inject_turn_badge,
    crop_viewbox,
)


class SegmentRenderer:
    def render_segment(
        self,
        base_svg: str,
        segment: RouteSegment,
        graph: HospitalGraph,
    ) -> str:
        """Render a single route segment onto the base SVG. Returns cropped SVG string."""
        points = [
            (graph.nodes[nid].x, graph.nodes[nid].y)
            for nid in segment.nodes
            if nid in graph.nodes
        ]
        if not points:
            return base_svg

        svg = inject_route_overlay(base_svg, points)

        # Detect turn points within this segment
        turn_indices = []
        for i in range(1, len(points) - 1):
            angle = angle_between(points[i - 1], points[i], points[i + 1])
            if abs(angle) >= 30:
                turn_indices.append(i)

        if turn_indices:
            svg = inject_arrows(svg, points, turn_indices)

        # Label landmarks
        labels = []
        for nid in segment.nodes:
            node = graph.nodes.get(nid)
            if node and node.type != "junction" and node.name:
                labels.append({"x": node.x, "y": node.y - 12, "text": node.name})
        if labels:
            svg = inject_labels(svg, labels)

        # Turn badge at the end of segment if not straight
        if segment.direction != "straight" and len(points) >= 2:
            svg = inject_turn_badge(svg, points[-1], segment.direction)

        bbox = bounding_box(points, padding=80.0)
        svg = crop_viewbox(svg, bbox)

        return svg

    def render_all_segments(
        self,
        base_svgs: dict[int, str],
        segments: list[RouteSegment],
        graph: HospitalGraph,
    ) -> list[str | None]:
        """Render all segments. base_svgs keyed by floor number.

        Returns one entry per segment. Entries are None when the floor SVG
        is unavailable so callers can match results to segments by index.
        """
        results: list[str | None] = []
        for segment in segments:
            floor_svg = base_svgs.get(segment.floor, "")
            if not floor_svg:
                results.append(None)
                continue
            rendered = self.render_segment(floor_svg, segment, graph)
            results.append(rendered)
        return results

    def render_full_route(
        self,
        base_svgs: dict[int, str],
        segments: list[RouteSegment],
        graph: HospitalGraph,
    ) -> dict[int, str]:
        """Render the entire route onto one SVG per floor.

        Returns {floor: svg_string}. All segments on the same floor are
        drawn onto a single SVG with one bounding-box crop covering the
        full route on that floor.
        """
        floors: dict[int, list[RouteSegment]] = {}
        for seg in segments:
            floors.setdefault(seg.floor, []).append(seg)

        results: dict[int, str] = {}
        for floor, floor_segs in floors.items():
            floor_svg = base_svgs.get(floor, "")
            if not floor_svg:
                continue

            all_points: list[tuple[float, float]] = []
            for seg in floor_segs:
                pts = [
                    (graph.nodes[nid].x, graph.nodes[nid].y)
                    for nid in seg.nodes
                    if nid in graph.nodes
                ]
                all_points.extend(pts)

            if not all_points:
                continue

            svg = inject_route_overlay(floor_svg, all_points)

            turn_indices = []
            for i in range(1, len(all_points) - 1):
                angle = angle_between(all_points[i - 1], all_points[i], all_points[i + 1])
                if abs(angle) >= 30:
                    turn_indices.append(i)
            if turn_indices:
                svg = inject_arrows(svg, all_points, turn_indices)

            labels = []
            seen_labels: set[str] = set()
            for seg in floor_segs:
                for nid in seg.nodes:
                    node = graph.nodes.get(nid)
                    if node and node.type != "junction" and node.name and node.name not in seen_labels:
                        seen_labels.add(node.name)
                        labels.append({"x": node.x, "y": node.y - 12, "text": node.name})
            if labels:
                svg = inject_labels(svg, labels)

            for seg in floor_segs:
                if seg.direction != "straight" and len(seg.nodes) >= 2:
                    end_node = graph.nodes.get(seg.nodes[-1])
                    if end_node:
                        svg = inject_turn_badge(svg, (end_node.x, end_node.y), seg.direction)

            bbox = bounding_box(all_points, padding=80.0)
            svg = crop_viewbox(svg, bbox)
            results[floor] = svg

        return results
