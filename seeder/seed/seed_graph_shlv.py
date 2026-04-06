import asyncio
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

SHLV_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "graphs", "shlv.json",
)


async def seed():
    from app.repositories.GraphRepository import graphRepository

    with open(SHLV_JSON_PATH, "r", encoding="utf-8") as f:
        doc = json.load(f)

    building_id = doc.pop("_id", "shlv")
    doc.pop("version", None)
    doc.pop("updated_by", None)
    doc.pop("updated_at", None)

    version = await graphRepository.save_graph(building_id, doc, updated_by="seed_graph_shlv")

    nodes = doc.get("nodes", [])
    rooms = sum(1 for n in nodes if n.get("type") != "junction")
    junctions = sum(1 for n in nodes if n.get("type") == "junction")

    logger.info(
        "Seeded %s: %d nodes total (%d rooms, %d junctions) across floors %s (version %d)",
        building_id,
        len(nodes),
        rooms,
        junctions,
        doc.get("floors", []),
        version,
    )


async def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    await seed()


if __name__ == "__main__":
    asyncio.run(main())
