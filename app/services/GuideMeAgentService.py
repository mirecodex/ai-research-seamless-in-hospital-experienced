import datetime
import json
import traceback
from typing import List
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, ToolMessage

from core.BaseAgent import BaseAgent
from app.tools.VirtualQueue import virtualQueueHandler
from app.tools.AISearchNavigate import aiSearchHandler
from app.tools.Pathfinding import pathfindingHandler
from core.navigation.prompt import GUIDE_ME_PROMPT


class GuideMeAgent(BaseAgent):

    def __init__(self, llm, **kwargs):
        @tool
        async def virtual_queue_lookup(queue_number: str, building_id: str = "shlv") -> str:
            """Look up a virtual queue number to find the patient's destination.
            Returns JSON with queue status, destination name, estimated wait time.
            Example queue numbers: 'B-045', 'A-012'.
            """
            result = await virtualQueueHandler.get_queue_destination(queue_number, building_id)
            return json.dumps(result, ensure_ascii=False)

        @tool
        async def ai_search_navigate(query: str, building_id: str = "shlv") -> str:
            """Resolve a location name to a node ID in the hospital graph.
            Use after getting the destination name from virtual_queue_lookup.
            Returns JSON with found (bool), node_id, name, floor.
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
            Returns JSON with success (bool), path, total_distance, estimated_time_seconds.
            """
            result = pathfindingHandler.execute(from_node, to_node, building_id, profile)
            return json.dumps(result, ensure_ascii=False)

        tools: List[BaseTool] = [virtual_queue_lookup, ai_search_navigate, pathfinding]

        super().__init__(
            llm=llm,
            prompt_template=GUIDE_ME_PROMPT,
            tools=tools,
            **kwargs
        )

    async def __call__(self, state):
        try:
            input_data = state.get("input", {})
            user_query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            building_id = state.get("building_id", "shlv")
            current_location = state.get("current_location", "f1_j1")

            self.rebind_prompt_variable(
                time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                building_id=building_id,
                current_location=current_location or "f1_j1",
            )

            messages = [HumanMessage(content=user_query)]
            agent_state = {"messages": messages}

            tool_calls_info = []
            route_data = None
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
                            if "building_id" not in tool_args and tool_name != "virtual_queue_lookup":
                                tool_args["building_id"] = building_id
                            if tool_name == "pathfinding" and "from_node" not in tool_args:
                                tool_args["from_node"] = current_location or "f1_j1"

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

                            messages.append(ToolMessage(
                                content=tool_result_str,
                                tool_call_id=tool_call["id"],
                            ))

                    agent_state = {"messages": messages}
                else:
                    final_answer = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
                    break

            if not final_answer:
                final_answer = raw_result.content if hasattr(raw_result, "content") else "Panduan antrian selesai."

            response_data = {
                "tool_calls": tool_calls_info,
                "final_answer": final_answer,
            }

            return {
                "response": json.dumps(response_data, indent=2, ensure_ascii=False),
                "input": state.get("input", {}),
                "route_data": route_data,
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
