from pydantic import BaseModel, Field
from typing import Optional


class NavigationRequest(BaseModel):
    query: str = Field(..., description="User message or destination query")
    building_id: str = Field(default="shlv", description="Target building ID")
    current_floor: Optional[int] = None
    current_location: Optional[str] = None
    profile: str = Field(default="default", description="Routing profile: default, wheelchair, elderly, emergency")
    output_format: str = Field(default="svg", description="Output format: svg or png")
    start_id: Optional[str] = None
    end_id: Optional[str] = None


class NavigationDirectRequest(BaseModel):
    from_node: str = Field(..., description="Starting node ID")
    to_node: str = Field(..., description="Destination node ID")
    profile: str = Field(default="default")
    building_id: str = Field(default="shlv")
    output_format: str = Field(default="svg", description="Output format: svg or png")
