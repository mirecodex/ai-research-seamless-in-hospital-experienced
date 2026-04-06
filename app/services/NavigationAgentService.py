import datetime
import json
import traceback
from typing import List
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, ToolMessage

from core.BaseAgent import BaseAgent
from app.tools.AISearchNavigate import aiSearchHandler
from app.tools.Pathfinding import pathfindingHandler
from app.tools.RouteRenderer import routeRendererHandler
from app.services.InstructionGenService import InstructionGenerator
from core.navigation.prompt import NAVIGATION_AGENT_PROMPT


class NavigationAgent(BaseAgent):

    def __init__(self, llm, **kwargs):
        self.instruction_gen = InstructionGenerator(llm=llm)

        @tool
        async def ai_search_navigate(query: str, building_id: str = "shlv") -> str:
            """Resolve a natural language location query to a specific node ID in the hospital.
            Searches node names and aliases. Handles Bahasa Indonesia and English.
            Example queries: 'toilet', 'farmasi', 'klinik 3', 'radiologi'.
            Returns JSON with found (bool), node_id, name, floor, aliases.
            """
            result = await aiSearchHandler.resolve(query, building_id)
            return json.dumps(result, ensure_ascii=False)

        @tool
        async def pathfinding(
            from_node: str,
            to_node: str,
            building_id: str = "shlv",
            profile: str = "default",
        ) -> str:
            """Find the shortest route between two nodes using A* pathfinding.
            Returns JSON with success (bool), path (list of node IDs), total_distance, estimated_time_seconds, steps, floors_visited.
            Use the node_id from ai_search_navigate as to_node.
            """
            result = pathfindingHandler.execute(from_node, to_node, building_id, profile)
            return json.dumps(result, ensure_ascii=False)

        @tool
        async def route_renderer(
            path: str,
            building_id: str = "shlv",
            profile: str = "default",
            output_format: str = "svg",
        ) -> str:
            """Render visual route images for each navigation segment.
            path MUST be a JSON array string of node IDs from pathfinding result, e.g. '["j1","j5","j9","f12a-..."]'.
            Returns JSON array with step, floor, direction, landmarks, distance_m per segment. Images are stored separately.
            """
            path_list = json.loads(path) if isinstance(path, str) else path
            full_result = await routeRendererHandler.render(path_list, building_id, profile, output_format)
            # Return only metadata to LLM (strip svg_data to avoid token explosion)
            summary = []
            for seg in full_result:
                summary.append({
                    "step": seg.get("step"),
                    "floor": seg.get("floor"),
                    "direction": seg.get("direction"),
                    "landmarks": seg.get("landmarks", []),
                    "distance_m": seg.get("distance_m", 0),
                    "floor_change": seg.get("floor_change"),
                    "has_image": bool(seg.get("svg_data") or seg.get("image_url")),
                })
            return json.dumps(summary, ensure_ascii=False)

        tools: List[BaseTool] = [ai_search_navigate, pathfinding, route_renderer]

        super().__init__(
            llm=llm,
            prompt_template=NAVIGATION_AGENT_PROMPT,
            tools=tools,
            **kwargs
        )

    async def __call__(self, state):
        try:
            input_data = state.get("input", {})
            user_query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            building_id = state.get("building_id", "shlv")
            current_location = state.get("current_location", "j1")
            current_floor = state.get("current_floor", 1)
            output_format = state.get("output_format", "svg")

            self.rebind_prompt_variable(
                time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                building_id=building_id,
                current_location=current_location or "j1",
                current_floor=str(current_floor or 1),
            )

            messages = [HumanMessage(content=user_query)]
            agent_state = {"messages": messages}

            tool_calls_info = []
            route_data = None
            rendered_images = None
            final_answer = None

            max_iterations = 8
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                raw_result, parsed_result = await self.arun_chain(state=agent_state)

                if hasattr(raw_result, "tool_calls") and raw_result.tool_calls:
                    messages.append(raw_result)

                    for tool_call in raw_result.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        tool_to_use = next(
                            (t for t in self.tools if t.name == tool_name), None
                        )

                        if tool_to_use:
                            if tool_name == "pathfinding":
                                if "building_id" not in tool_args:
                                    tool_args["building_id"] = building_id
                                tool_args["from_node"] = current_location or "j1"
                            if tool_name == "route_renderer":
                                if "building_id" not in tool_args:
                                    tool_args["building_id"] = building_id
                                if "output_format" not in tool_args:
                                    tool_args["output_format"] = output_format
                            if tool_name == "ai_search_navigate":
                                if "building_id" not in tool_args:
                                    tool_args["building_id"] = building_id

                            tool_result = await tool_to_use.ainvoke(tool_args)
                            tool_result_str = str(tool_result)

                            tool_calls_info.append({
                                "tool": tool_name,
                                "args": tool_args,
                                "result_preview": tool_result_str[:300],
                            })

                            if tool_name == "pathfinding":
                                try:
                                    parsed = json.loads(tool_result_str)
                                    if parsed.get("success"):
                                        route_data = parsed
                                except json.JSONDecodeError:
                                    pass
                            elif tool_name == "route_renderer":
                                path_arg = tool_args.get("path", "[]")
                                path_list = json.loads(path_arg) if isinstance(path_arg, str) else path_arg
                                rendered_images = await routeRendererHandler.render(
                                    path_list,
                                    tool_args.get("building_id", building_id),
                                    tool_args.get("profile", "default"),
                                    tool_args.get("output_format", output_format),
                                )

                            messages.append(ToolMessage(
                                content=tool_result_str,
                                tool_call_id=tool_call["id"],
                            ))

                    agent_state = {"messages": messages}
                else:
                    final_answer = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
                    break

            if not final_answer:
                final_answer = raw_result.content if hasattr(raw_result, "content") else "Navigasi selesai."

            instructions = []
            if rendered_images and isinstance(rendered_images, list):
                instructions = await self._generate_instructions(rendered_images)

            response_data = {
                "tool_calls": tool_calls_info,
                "final_answer": final_answer,
            }

            return {
                "response": json.dumps(response_data, indent=2, ensure_ascii=False),
                "input": state.get("input", {}),
                "route_data": route_data,
                "rendered_images": rendered_images,
                "instructions": instructions,
            }
        except Exception as e:
            traceback.print_exc()
            error_response = {
                "error": str(e),
                "tool_calls": [],
                "final_answer": None,
            }
            return {
                "response": json.dumps(error_response, indent=2, ensure_ascii=False),
                "input": state.get("input", {}),
            }

    async def _generate_instructions(self, rendered_segments: list) -> list[str]:
        instructions = []
        for seg in rendered_segments:
            direction = seg.get("direction", "straight")
            distance_m = seg.get("distance_m", 0)
            landmarks = seg.get("landmarks", [])
            floor = seg.get("floor", 1)
            floor_change = seg.get("floor_change")

            try:
                instruction = await self.instruction_gen.generate(
                    direction=direction,
                    distance_m=distance_m,
                    landmarks=landmarks,
                    floor=floor,
                    floor_change=floor_change,
                )
                if not instruction or not instruction.strip():
                    instruction = self._fallback_instruction(
                        direction, distance_m, landmarks, floor_change
                    )
            except Exception:
                instruction = self._fallback_instruction(
                    direction, distance_m, landmarks, floor_change
                )

            instructions.append(instruction)
        return instructions

    @staticmethod
    def _fallback_instruction(
        direction: str,
        distance_m: float,
        landmarks: list[str],
        floor_change: dict | None,
    ) -> str:
        if floor_change:
            from_f = floor_change.get("from_floor", "?")
            to_f = floor_change.get("to_floor", "?")
            via = floor_change.get("via", "lift")
            via_label = {"elevator": "lift", "stairs": "tangga"}.get(via, via)
            verb = "Naik" if (isinstance(to_f, int) and isinstance(from_f, int) and to_f > from_f) else "Turun"
            return f"{verb} {via_label} dari Lantai {from_f} ke Lantai {to_f}."

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
        steps = max(1, int(distance_m / 0.7)) if distance_m > 0 else 5
        landmark_str = ", ".join(landmarks) if landmarks else "tujuan"
        return f"Jalan {dir_str} sekitar {steps} langkah menuju {landmark_str}."
