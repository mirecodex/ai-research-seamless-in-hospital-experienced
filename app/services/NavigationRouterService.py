import datetime
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from core.BaseAgent import BaseAgent
from app.schemas.NavigationRouterOutputSchema import NavigationRouterOutput
from core.navigation.prompt import NAVIGATION_ROUTER_PROMPT


class NavigationRouter(BaseAgent):

    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=NAVIGATION_ROUTER_PROMPT,
            output_model=NavigationRouterOutput,
            use_structured_output=False,
            **kwargs
        )

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.rebind_prompt_variable(
            time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        input_data = state.get("input", {})
        user_query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)

        agent_state = {"messages": [HumanMessage(content=user_query)]}
        raw, parsed = await self.arun_chain(state=agent_state)

        decision = parsed.get("intent", "fallback") if isinstance(parsed, dict) else parsed.intent.value
        reasoning = parsed.get("reasoning", "") if isinstance(parsed, dict) else parsed.reasoning

        return {
            "decision": decision,
            "input": state.get("input", {}),
            "building_id": state.get("building_id", "shlv"),
            "current_location": state.get("current_location"),
            "current_floor": state.get("current_floor"),
            "output_format": state.get("output_format", "svg"),
            "response": f"Routing to {decision}. Reasoning: {reasoning}",
        }
