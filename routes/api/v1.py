import app.schemas as schemas
from fastapi import APIRouter, Query, Request
# from config.manual_mcp import mcpconfig
from app.controllers.ChatbotController import chatbotController
from app.controllers.NavigationController import navigationController
from app.schemas.NavigationInputSchema import NavigationRequest, NavigationDirectRequest
from app.schemas.GraphCrudSchema import GraphImportPayload
from app.repositories.GraphRepository import graphRepository
from core.navigation import GraphManager, find_route
from app.utils.HttpResponseUtils import response_success, response_error, response_format
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os

router = APIRouter()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@router.post("/test")
async def endpoint_test(req: Request, input: schemas.Item):
    return await chatbotController.start_chatting(input)


# --- Navigation ---

@router.post("/navigate")
async def navigate(input: NavigationRequest):
    return await navigationController.start_navigating(input.model_dump())


@router.post("/route")
async def direct_route(input: NavigationDirectRequest):
    try:
        graph = GraphManager.get(input.building_id)
        if not graph:
            return response_format("Building not found", 404)
        result = find_route(graph, input.from_node, input.to_node, input.profile)
        return response_success(result.model_dump())
    except Exception as e:
        return response_error(str(e))


@router.post("/navigate/direct")
async def navigate_direct(input: NavigationDirectRequest):
    return await navigationController.navigate_direct(input.model_dump())


# --- Graph Admin ---

@router.post("/graph/import")
async def import_graph(payload: GraphImportPayload):
    try:
        doc = {
            "building_name": payload.building_name,
            "floors": payload.floors,
            "nodes": payload.nodes,
        }
        version = await graphRepository.save_graph(payload.building_id, doc, updated_by="api")
        await GraphManager.reload(payload.building_id)
        return response_success({"building_id": payload.building_id, "version": version})
    except Exception as e:
        return response_error(str(e))


@router.get("/graph/export/{building_id}")
async def export_graph(building_id: str):
    try:
        doc = await graphRepository.get_graph(building_id)
        if not doc:
            return response_format("Building not found", 404)
        return response_success(doc)
    except Exception as e:
        return response_error(str(e))


@router.get("/rooms/sync")
async def sync_rooms(building_id: str = "shlv"):
    try:
        rooms = await graphRepository.get_rooms(building_id)
        doc = await graphRepository.get_graph(building_id)
        version = doc.get("version", 0) if doc else 0
        return response_success({
            "building_id": building_id,
            "version": version,
            "rooms": rooms,
        })
    except Exception as e:
        return response_error(str(e))


@router.get("/buildings")
async def list_buildings():
    return response_success(GraphManager.list_buildings())


@router.get("/buildings/{building_id}/graph")
async def get_building_graph(building_id: str):
    try:
        doc = await graphRepository.get_graph(building_id)
        if not doc:
            return response_format("Building not found", 404)
        return response_success(doc)
    except Exception as e:
        return response_error(str(e))


@router.get("/locations")
async def list_locations(building_id: str = "shlv"):
    graph = GraphManager.get(building_id)
    if not graph:
        return response_format("Building not found", 404)
    return response_success(graph.get_locations())


@router.get("/locations/search")
async def search_locations(
    q: str = Query(..., min_length=1, description="Search query"),
    building_id: str = Query(default="shlv", description="Building ID"),
    max_results: int = Query(default=10, ge=1, le=50, description="Maximum results"),
):
    graph = GraphManager.get(building_id)
    if not graph:
        return response_format("Building not found", 404)
    nodes = graph.search_locations(q, max_results=max_results)
    results = [
        {
            "id": node.id,
            "name": node.name,
            "floor": node.floor,
            "category": node.category,
            "aliases": node.aliases,
        }
        for node in nodes
    ]
    return response_success(results)


@router.get("/floors/{building_id}/{floor}.svg")
async def get_floor_svg(building_id: str, floor: int):
    svg_path = os.path.join(_PROJECT_ROOT, "data", "floors", building_id, f"{floor}.svg")
    if not os.path.exists(svg_path):
        return response_format("Floor SVG not found", 404)
    return FileResponse(svg_path, media_type="image/svg+xml")


# --- Floor SVG Upload ---

class SvgUploadPayload(BaseModel):
    svg_content: str = Field(..., description="Raw SVG content string")
    floor_number: int = Field(default=1, description="Floor number")


@router.post("/floors/{building_id}/svg")
async def upload_floor_svg(building_id: str, payload: SvgUploadPayload):
    try:
        floor_dir = os.path.join(_PROJECT_ROOT, "data", "floors", building_id)
        os.makedirs(floor_dir, exist_ok=True)

        svg_path = os.path.join(floor_dir, f"{payload.floor_number}.svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(payload.svg_content)

        url = f"/api/v1/floors/{building_id}/{payload.floor_number}.svg"
        return response_success({"url": url, "building_id": building_id, "floor": payload.floor_number})
    except Exception as e:
        return response_error(str(e))