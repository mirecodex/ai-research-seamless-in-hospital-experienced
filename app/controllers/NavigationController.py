import json
import traceback
import uuid

from app.generative import manager as AiManager
from app.utils.HttpResponseUtils import response_success, response_error, response_format
from app.services.NavigationRouterService import NavigationRouter
from app.services.NavigationAgentService import NavigationAgent
from app.services.GuideMeAgentService import GuideMeAgent
from app.services.GraphInfoAgentService import GraphInfoAgent
from app.services.InstructionGenService import InstructionGenerator
from app.schemas.NavigationStateSchema import NavigationState
from app.schemas.WebSocketMessageSchema import WSRouteMeta, WSRouteResult, WSRouteFloorImage, WSRouteComplete, WSError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from core.navigation import GraphManager, find_route
from app.tools.RouteRenderer import routeRendererHandler
from typing import Dict


class NavigationController:

    def __init__(self):
        self.llm = AiManager.gemini_mini()
        self.router = NavigationRouter(llm=self.llm)
        self.nav_agent = NavigationAgent(llm=self.llm)
        self.guide_me_agent = GuideMeAgent(llm=self.llm)
        self.graph_info_agent = GraphInfoAgent(llm=self.llm)
        self.instruction_gen = InstructionGenerator(llm=self.llm)
        self.build_graph(checkpoint=InMemorySaver())

    async def nav_agent_node(self, state: NavigationState) -> Dict:
        try:
            result = await self.nav_agent(state)
            return result
        except Exception as e:
            traceback.print_exc()
            return {"response": json.dumps({"error": str(e)})}

    async def info_node(self, state: NavigationState) -> Dict:
        try:
            result = await self.graph_info_agent(state)
            return result
        except Exception as e:
            traceback.print_exc()
            return {"response": json.dumps({"error": str(e)})}

    async def guide_me_node(self, state: NavigationState) -> Dict:
        try:
            result = await self.guide_me_agent(state)
            return result
        except Exception as e:
            traceback.print_exc()
            return {"response": json.dumps({"error": str(e)})}

    async def fallback_node(self, state: NavigationState) -> Dict:
        return {
            "response": json.dumps({
                "final_answer": "Maaf, saya hanya bisa membantu navigasi dan informasi fasilitas rumah sakit. Silakan tanyakan arah ke suatu lokasi atau informasi fasilitas."
            }),
            "input": state.get("input", {}),
        }

    def build_graph(self, checkpoint=None):
        workflow = StateGraph(NavigationState)

        workflow.add_node("router", self.router)
        workflow.add_node("nav_agent", self.nav_agent_node)
        workflow.add_node("guide_me", self.guide_me_node)
        workflow.add_node("graph_info", self.info_node)
        workflow.add_node("fallback", self.fallback_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            lambda state: state.get("decision", "fallback"),
            {
                "navigation": "nav_agent",
                "guide_me": "guide_me",
                "info": "graph_info",
                "fallback": "fallback",
            }
        )

        workflow.add_edge("nav_agent", END)
        workflow.add_edge("guide_me", END)
        workflow.add_edge("graph_info", END)
        workflow.add_edge("fallback", END)

        graph = workflow.compile(checkpointer=checkpoint)
        self.graph = graph
        return graph

    # -- Shared helpers --

    def _combine_instructions(self, segments: list, instructions: list) -> str:
        lines = []
        for i, seg in enumerate(segments):
            text = instructions[i] if i < len(instructions) else ""
            if not text or not text.strip():
                text = self._fallback_step_instruction(seg)
            lines.append(text)
        return "\n".join(lines)

    def _build_full_response(
        self,
        route_data: dict,
        full_render: dict,
        combined_instruction: str,
        all_landmarks: list[str],
    ) -> dict:
        images = []
        for floor, img in full_render.get("floors", {}).items():
            images.append({
                "floor": int(floor),
                "svg_data": img.get("svg_data"),
                "image_url": img.get("image_url"),
            })
        images.sort(key=lambda x: x["floor"])

        return {
            "route_data": route_data,
            "images": images,
            "instruction": combined_instruction,
            "landmarks": all_landmarks,
        }

    # -- HTTP endpoints --

    async def start_navigating(self, input_data: dict):
        try:
            initial_state = NavigationState(
                input=input_data,
                decision="",
                building_id=input_data.get("building_id", "shlv"),
                current_location=input_data.get("current_location"),
                current_floor=input_data.get("current_floor"),
                output_format=input_data.get("output_format", "svg"),
                route_data=None,
                segments=None,
                rendered_images=None,
                instructions=None,
                response="",
            )
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            result = await self.graph.ainvoke(initial_state, config=config)

            route_data = self._parse_json_field(result.get("route_data"))
            rendered = self._parse_json_field(result.get("rendered_images")) or []
            instructions = result.get("instructions") or []

            if route_data and route_data.get("success") and rendered:
                path = route_data.get("path") or route_data.get("nodes_visited") or []
                building_id = input_data.get("building_id", "shlv")
                output_format = input_data.get("output_format", "svg")

                full_render = await routeRendererHandler.render_full(
                    path, building_id, "default", output_format,
                )
                combined = self._combine_instructions(rendered, instructions)
                all_landmarks = []
                seen = set()
                for seg in rendered:
                    for lm in seg.get("landmarks", []):
                        if lm not in seen:
                            seen.add(lm)
                            all_landmarks.append(lm)

                payload = self._build_full_response(route_data, full_render, combined, all_landmarks)
                return response_success(payload)

            agent_response = result.get("response", "")
            agent_msg = ""
            try:
                parsed = json.loads(agent_response)
                agent_msg = parsed.get("final_answer") or parsed.get("error") or ""
            except (json.JSONDecodeError, TypeError):
                agent_msg = agent_response

            query = input_data.get("query", "tujuan")
            error_code, fallback_msg = self._classify_nav_error(result, agent_msg, query)
            return response_format(agent_msg or fallback_msg, 422)
        except Exception as e:
            traceback.print_exc()
            return response_error(e)

    async def navigate_direct(self, input_data: dict):
        building_id = input_data.get("building_id", "shlv")
        from_node = input_data.get("from_node", "")
        to_node = input_data.get("to_node", "")
        profile = input_data.get("profile", "default")
        output_format = input_data.get("output_format", "svg")

        try:
            graph = GraphManager.get(building_id)
            if not graph:
                return response_format("Building not found", 404)

            route_response = find_route(graph, from_node, to_node, profile)
            if not route_response.success:
                return response_format(route_response.error or "Route not found", 422)

            path = route_response.nodes_visited
            full_render = await routeRendererHandler.render_full(path, building_id, profile, output_format)
            segments = full_render.get("segments", [])

            instructions = []
            for seg in segments:
                try:
                    instruction = await self.instruction_gen.generate(
                        direction=seg.get("direction", "straight"),
                        distance_m=seg.get("distance_m", 0),
                        landmarks=seg.get("landmarks", []),
                        floor=seg.get("floor", 1),
                        floor_change=seg.get("floor_change"),
                    )
                    if not instruction or not instruction.strip():
                        instruction = self._fallback_step_instruction(seg)
                except Exception:
                    instruction = self._fallback_step_instruction(seg)
                instructions.append(instruction)

            combined = "\n".join(instructions)
            all_landmarks = []
            seen = set()
            for seg in segments:
                for lm in seg.get("landmarks", []):
                    if lm not in seen:
                        seen.add(lm)
                        all_landmarks.append(lm)

            route_data = {
                "success": True,
                "total_distance": route_response.total_distance,
                "estimated_time_seconds": route_response.estimated_time_seconds,
                "floors_visited": list({s.get("floor", 1) for s in segments}),
                "path": path,
            }

            payload = self._build_full_response(route_data, full_render, combined, all_landmarks)
            return response_success(payload)
        except Exception as e:
            traceback.print_exc()
            return response_error(e)

    # -- Utilities --

    def _parse_json_field(self, value):
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    @staticmethod
    def _fallback_step_instruction(seg: dict) -> str:
        floor_change = seg.get("floor_change")
        if floor_change:
            from_f = floor_change.get("from_floor", "?")
            to_f = floor_change.get("to_floor", "?")
            via = floor_change.get("via", "lift")
            via_label = {"elevator": "lift", "stairs": "tangga"}.get(via, via)
            verb = "Naik" if (isinstance(to_f, int) and isinstance(from_f, int) and to_f > from_f) else "Turun"
            return f"{verb} {via_label} dari Lantai {from_f} ke Lantai {to_f}."

        direction = seg.get("direction", "straight")
        direction_labels = {
            "straight": "lurus",
            "right": "belok kanan",
            "left": "belok kiri",
            "slight_right": "agak ke kanan",
            "slight_left": "agak ke kiri",
            "sharp_right": "belok kanan tajam",
            "sharp_left": "belok kiri tajam",
        }
        dir_str = direction_labels.get(direction, "lurus")
        distance_m = seg.get("distance_m", 0)
        steps = max(1, int(distance_m / 0.7)) if distance_m > 0 else 5
        landmarks = seg.get("landmarks", [])
        landmark_str = ", ".join(landmarks) if landmarks else "tujuan"
        return f"Jalan {dir_str} sekitar {steps} langkah menuju {landmark_str}."

    @staticmethod
    def _classify_nav_error(result: dict, agent_msg: str, query: str) -> tuple[str, str]:
        lower = agent_msg.lower()
        route_data = result.get("route_data")

        if route_data and not route_data.get("success"):
            return "ROUTE_FAILED", f"Tidak dapat menemukan rute menuju '{query}'."

        loc_keywords = ("tidak ditemukan", "not found", "lokasi", "lokasi tidak", "tidak bisa menemukan")
        if any(kw in lower for kw in loc_keywords):
            return "LOC_NOT_FOUND", f"Lokasi '{query}' tidak ditemukan."

        return "ROUTE_FAILED", f"Tidak dapat menemukan rute menuju '{query}'."

    # -- WebSocket handler --

    async def handle_websocket(self, websocket):
        from starlette.websockets import WebSocketState
        await websocket.accept()

        try:
            while True:
                data = await websocket.receive_json()
                correlation_id = str(uuid.uuid4())

                try:
                    initial_state = NavigationState(
                        input=data,
                        decision="",
                        building_id=data.get("building_id", "shlv"),
                        current_location=data.get("current_location"),
                        current_floor=data.get("current_floor"),
                        output_format=data.get("output_format", "svg"),
                        route_data=None,
                        segments=None,
                        rendered_images=None,
                        instructions=None,
                        response="",
                    )
                    config = {"configurable": {"thread_id": correlation_id}}
                    result = await self.graph.ainvoke(initial_state, config=config)

                    route_data = self._parse_json_field(result.get("route_data"))
                    rendered = self._parse_json_field(result.get("rendered_images")) or []
                    instructions = result.get("instructions") or []

                    if route_data and route_data.get("success"):
                        if not rendered:
                            query = data.get("query", "tujuan")
                            await websocket.send_json(WSError(
                                code="RENDER_FAILED",
                                message=f"Gagal merender rute menuju '{query}'. Silakan coba lagi.",
                            ).model_dump())
                            continue

                        path = route_data.get("path") or route_data.get("nodes_visited") or []
                        building_id = data.get("building_id", "shlv")
                        output_format = data.get("output_format", "svg")

                        full_render = await routeRendererHandler.render_full(
                            path, building_id, "default", output_format,
                        )

                        floors_visited = route_data.get("floors_visited", [])
                        meta = WSRouteMeta(
                            total_distance_m=route_data.get("total_distance", 0),
                            estimated_time_s=int(route_data.get("estimated_time_seconds", 0)),
                            floors_involved=floors_visited,
                            correlation_id=correlation_id,
                        )
                        await websocket.send_json(meta.model_dump())

                        combined = self._combine_instructions(rendered, instructions)
                        all_landmarks = []
                        seen = set()
                        for seg in rendered:
                            for lm in seg.get("landmarks", []):
                                if lm not in seen:
                                    seen.add(lm)
                                    all_landmarks.append(lm)

                        images = []
                        for floor, img in full_render.get("floors", {}).items():
                            images.append(WSRouteFloorImage(
                                floor=int(floor),
                                svg_data=img.get("svg_data"),
                                image_url=img.get("image_url"),
                            ))
                        images.sort(key=lambda x: x.floor)

                        route_result = WSRouteResult(
                            images=images,
                            instruction=combined,
                            landmarks=all_landmarks,
                        )
                        await websocket.send_json(route_result.model_dump())

                        dest_name = data.get("query", "tujuan")
                        complete = WSRouteComplete(
                            destination=dest_name,
                            message=f"Anda telah sampai di {dest_name}.",
                        )
                        await websocket.send_json(complete.model_dump())
                    else:
                        response_text = result.get("response") or ""
                        agent_msg = ""
                        try:
                            parsed = json.loads(response_text)
                            agent_msg = parsed.get("final_answer") or parsed.get("error") or ""
                        except (json.JSONDecodeError, TypeError):
                            agent_msg = response_text

                        query = data.get("query", "")
                        error_code, fallback_msg = self._classify_nav_error(result, agent_msg, query)
                        await websocket.send_json(WSError(
                            code=error_code,
                            message=agent_msg or fallback_msg,
                        ).model_dump())

                except Exception as e:
                    traceback.print_exc()
                    await websocket.send_json(WSError(
                        code="INTERNAL_ERROR",
                        message=str(e),
                    ).model_dump())

        except Exception:
            pass
        finally:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()


navigationController = NavigationController()
