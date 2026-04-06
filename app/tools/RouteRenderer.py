import hashlib
import logging
import os
from typing import Optional

from core.navigation.manager import GraphManager
from core.navigation.segmenter import RouteSegmenter, RouteSegment
from core.navigation.renderer import SegmentRenderer
from core.navigation.pathfinding import MAP_RATIO

logger = logging.getLogger(__name__)

# Resolve project root from this file: app/tools/RouteRenderer.py -> project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class RouteRendererHandler:
    """Segments a route and renders visual overlays.

    SVG output for web clients. Playwright->PNG->S3 for WhatsApp.
    """

    def __init__(self):
        self._segmenter = RouteSegmenter()
        self._renderer = SegmentRenderer()
        self._floor_svgs: dict[str, dict[int, str]] = {}

    async def render(
        self,
        path: list[str],
        building_id: str = "shlv",
        profile: str = "default",
        output_format: str = "svg",
    ) -> list[dict]:
        """Segment route and render images.

        Returns list of dicts per segment with svg_data and/or image_url.
        """
        graph = GraphManager.get(building_id)
        if not graph:
            logger.warning("RouteRenderer: building '%s' not loaded", building_id)
            return []

        segments = self._segmenter.segment(graph, path, profile)
        if not segments:
            logger.warning("RouteRenderer: segmenter returned no segments for path of %d nodes", len(path))
            return []

        base_svgs = await self._get_floor_svgs(building_id, graph)
        if not base_svgs:
            logger.warning(
                "RouteRenderer: no floor SVGs loaded for building '%s'. "
                "Checked floors: %s. Returning segments without images.",
                building_id, graph.floors,
            )

        # render_all_segments now returns one entry per segment (None if floor SVG missing)
        svg_strings = self._renderer.render_all_segments(base_svgs, segments, graph)

        results = []
        for i, (segment, svg_str) in enumerate(zip(segments, svg_strings)):
            entry = {
                "step": i + 1,
                "floor": segment.floor,
                "direction": segment.direction,
                "landmarks": segment.landmarks,
                "distance_m": round(segment.distance / MAP_RATIO, 1),
                "floor_change": segment.floor_change,
                "svg_data": svg_str if output_format == "svg" and svg_str else None,
                "image_url": None,
            }

            if output_format == "png" and svg_str:
                url = await self._render_png_and_upload(svg_str, building_id, path, i)
                entry["image_url"] = url

            results.append(entry)

        return results

    async def render_full(
        self,
        path: list[str],
        building_id: str = "shlv",
        profile: str = "default",
        output_format: str = "svg",
    ) -> dict:
        """Render full route as one image per floor + combined segment metadata.

        Returns {"floors": {floor: {svg_data, image_url}}, "segments": [...]}
        """
        graph = GraphManager.get(building_id)
        if not graph:
            return {"floors": {}, "segments": []}

        segments = self._segmenter.segment(graph, path, profile)
        if not segments:
            return {"floors": {}, "segments": []}

        base_svgs = await self._get_floor_svgs(building_id, graph)
        floor_svgs = self._renderer.render_full_route(base_svgs, segments, graph)

        floors_result = {}
        for floor, svg_str in floor_svgs.items():
            entry = {"svg_data": svg_str if output_format == "svg" else None, "image_url": None}
            if output_format == "png" and svg_str:
                url = await self._render_png_and_upload(svg_str, building_id, path, floor)
                entry["image_url"] = url
            floors_result[floor] = entry

        seg_meta = []
        for i, seg in enumerate(segments):
            seg_meta.append({
                "step": i + 1,
                "floor": seg.floor,
                "direction": seg.direction,
                "landmarks": seg.landmarks,
                "distance_m": round(seg.distance / MAP_RATIO, 1),
                "floor_change": seg.floor_change,
            })

        return {"floors": floors_result, "segments": seg_meta}

    async def _get_floor_svgs(self, building_id: str, graph) -> dict[int, str]:
        """Load floor SVGs from local filesystem with caching."""
        if building_id in self._floor_svgs:
            return self._floor_svgs[building_id]

        svgs: dict[int, str] = {}

        for floor in graph.floors:
            svg_str = self._load_local_svg(building_id, floor)
            if svg_str:
                svgs[floor] = svg_str
                logger.info("Loaded floor SVG: %s/floor %d (%d bytes)", building_id, floor, len(svg_str))
            else:
                logger.warning("Floor SVG not found: %s/floor %d", building_id, floor)

        if svgs:
            self._floor_svgs[building_id] = svgs

        return svgs

    def _load_local_svg(self, building_id: str, floor: int) -> Optional[str]:
        candidates = [
            os.path.join(_PROJECT_ROOT, "data", "floors", building_id, f"{floor}.svg"),
            os.path.join(_PROJECT_ROOT, "data", "floors", building_id, f"LT{floor}.svg"),
        ]
        for filepath in candidates:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read()
        logger.debug("No SVG found at: %s", ", ".join(candidates))
        return None

    async def _render_png_and_upload(
        self, svg_str: str, building_id: str, path: list[str], step_idx: int
    ) -> Optional[str]:
        """Render SVG to PNG via Playwright and upload to S3."""
        try:
            from core.playwright import PlaywrightManager
            engine = await PlaywrightManager.acquire()
            if not engine:
                logger.warning("No Playwright engine available for PNG rendering")
                return None

            try:
                png_bytes = await engine.render_svg_to_png(svg_str)
            finally:
                PlaywrightManager.release(engine)

            path_hash = hashlib.md5(
                f"{building_id}:{'->'.join(path)}".encode()
            ).hexdigest()[:12]
            object_key = f"navigation/rendered/{path_hash}_step{step_idx + 1}.png"

            from app.traits.Uploader.S3UploaderUtils import S3Uploader
            # TODO: configure bucket name from settings
            bucket = "navigation-assets"
            success = S3Uploader.put_object(png_bytes, bucket, object_key)
            if success:
                return f"s3://{bucket}/{object_key}"
            return None

        except ImportError:
            logger.warning("Playwright or S3 not available for PNG rendering")
            return None
        except Exception as e:
            logger.error("PNG render/upload failed: %s", e)
            return None


routeRendererHandler = RouteRendererHandler()
