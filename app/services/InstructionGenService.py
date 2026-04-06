import datetime
from typing import Optional

from langchain_core.messages import HumanMessage

from core.BaseAgent import BaseAgent
from core.navigation.prompt import INSTRUCTION_GEN_PROMPT


class InstructionGenerator(BaseAgent):

    def __init__(self, llm, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=INSTRUCTION_GEN_PROMPT,
            **kwargs
        )

    async def generate(
        self,
        direction: str,
        distance_m: float,
        landmarks: list[str],
        floor: int,
        floor_change: Optional[dict] = None,
    ) -> str:
        distance_steps = int(distance_m / 0.7) if distance_m > 0 else 5

        landmarks_str = ", ".join(landmarks) if landmarks else "tidak ada landmark terdekat"
        floor_change_str = "tidak ada"
        if floor_change:
            fc_from = floor_change.get("from_floor") or floor_change.get("from", "?")
            fc_to = floor_change.get("to_floor") or floor_change.get("to", "?")
            fc_via = floor_change.get("via", "lift")
            floor_change_str = f"dari Lantai {fc_from} ke Lantai {fc_to} via {fc_via}"

        direction_labels = {
            "straight": "lurus",
            "right": "belok kanan",
            "left": "belok kiri",
            "slight_right": "agak kanan",
            "slight_left": "agak kiri",
            "sharp_right": "belok kanan tajam",
            "sharp_left": "belok kiri tajam",
        }
        direction_str = direction_labels.get(direction, direction)

        self.rebind_prompt_variable(
            time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            direction=direction_str,
            distance_steps=str(distance_steps),
            landmarks=landmarks_str,
            floor=str(floor),
            floor_change=floor_change_str,
        )

        agent_state = {"messages": [HumanMessage(content="Generate instruction")]}
        raw, parsed = await self.arun_chain(state=agent_state)

        if hasattr(raw, "content"):
            return raw.content.strip()
        return str(parsed).strip()

    async def __call__(self, state):
        segments = state.get("rendered_images", [])
        instructions = []

        for seg in segments:
            instruction = await self.generate(
                direction=seg.get("direction", "straight"),
                distance_m=seg.get("distance_m", 0),
                landmarks=seg.get("landmarks", []),
                floor=seg.get("floor", 1),
                floor_change=seg.get("floor_change"),
            )
            instructions.append(instruction)

        return {
            "instructions": instructions,
            "input": state.get("input", {}),
        }
