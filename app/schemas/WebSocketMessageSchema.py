from pydantic import BaseModel
from typing import Optional


class WSRouteMeta(BaseModel):
    type: str = "route_meta"
    total_distance_m: float
    estimated_time_s: int
    floors_involved: list[int]
    correlation_id: str = ""


class WSRouteFloorImage(BaseModel):
    floor: int
    svg_data: Optional[str] = None
    image_url: Optional[str] = None


class WSRouteResult(BaseModel):
    type: str = "route_result"
    images: list[WSRouteFloorImage] = []
    instruction: str = ""
    landmarks: list[str] = []


class WSRouteComplete(BaseModel):
    type: str = "route_complete"
    destination: str = ""
    message: str = ""


class WSError(BaseModel):
    type: str = "error"
    code: str = "INTERNAL_ERROR"
    message: str = ""
