import math
from typing import Optional

from .models import NodeData, EdgeData


class HospitalGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeData] = {}
        self.adjacency: dict[str, list[tuple[str, float, EdgeData]]] = {}
        self.building_id: str = ""
        self.building_name: str = ""
        self.floors: list[int] = []
        self._edges: list[EdgeData] = []

    # -- Loading from MongoDB document --

    @classmethod
    def from_mongo_doc(cls, doc: dict) -> "HospitalGraph":
        """Build graph from a MongoDB graph_data document.

        Expects: _id, building_name, floors, nodes[].
        Nodes have cx/cy coords, connection[] adjacency, objectName/categoryId/label
        for rooms. Edges are derived from connections, distances calculated at load time.
        """
        g = cls()
        g.building_id = doc.get("_id", "")
        g.building_name = doc.get("building_name", "")
        g.floors = [int(f) for f in doc.get("floors", [])]

        raw_nodes = doc.get("nodes", [])

        for raw in raw_nodes:
            node_type = raw.get("type", "junction")
            floor_val = raw.get("floor", 1)
            if isinstance(floor_val, str):
                floor_val = int(floor_val)

            node = NodeData(
                id=raw["id"],
                name=raw.get("objectName", ""),
                floor=floor_val,
                x=float(raw.get("cx", 0)),
                y=float(raw.get("cy", 0)),
                type=node_type,
                accessible=raw.get("accessible", True),
                aliases=raw.get("aliases", []),
                category=raw.get("categoryId", ""),
                description=raw.get("label", ""),
                metadata={
                    k: v for k, v in raw.items()
                    if k not in {
                        "id", "type", "floor", "cx", "cy", "connection",
                        "objectName", "categoryId", "label", "accessible",
                        "aliases", "keywords",
                    }
                },
            )
            g.nodes[node.id] = node
            g.adjacency[node.id] = []

        # Derive edges from connection[] fields
        seen_edges: set[tuple[str, str]] = set()
        for raw in raw_nodes:
            from_id = raw["id"]
            for to_id in raw.get("connection", []):
                if to_id not in g.nodes:
                    continue
                pair = tuple(sorted((from_id, to_id)))
                if pair in seen_edges:
                    continue
                seen_edges.add(pair)

                dist = g._euclidean(g.nodes[from_id], g.nodes[to_id])
                edge = EdgeData.model_validate({
                    "from": from_id,
                    "to": to_id,
                    "distance": round(dist, 1),
                })
                g._edges.append(edge)
                g.adjacency[from_id].append((to_id, edge.distance, edge))
                g.adjacency[to_id].append((from_id, edge.distance, edge))

        return g

    # -- Loading from spatial_data + base_data (editor export format) --

    @classmethod
    def from_editor_data(
        cls,
        spatial_data: list[dict],
        base_data: list[dict],
        building_id: str = "",
        building_name: str = "",
        floor: int = 1,
    ) -> "HospitalGraph":
        """Build graph from raw editor JSON exports (per-floor files).

        Merges base_data metadata into spatial_data nodes, then delegates
        to from_mongo_doc format.
        """
        base_lookup = {item["id"]: item for item in base_data}
        merged_nodes = []

        for spatial in spatial_data:
            node_id = spatial["id"]
            base = base_lookup.get(node_id, {})

            floor_val = base.get("floor", str(floor))
            if isinstance(floor_val, str):
                floor_val = int(floor_val) if floor_val.isdigit() else floor

            aliases_raw = base.get("aliases", "")
            if isinstance(aliases_raw, str):
                aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]
            else:
                aliases = aliases_raw

            merged = {
                "id": node_id,
                "type": spatial.get("type", "junction"),
                "floor": floor_val,
                "cx": spatial.get("cx", 0),
                "cy": spatial.get("cy", 0),
                "connection": spatial.get("connection", []),
                "objectName": base.get("label", ""),
                "categoryId": base.get("room-type", ""),
                "label": base.get("description", ""),
                "aliases": aliases,
                "accessible": True,
            }
            merged_nodes.append(merged)

        floors = sorted({n.get("floor", 1) for n in merged_nodes})

        doc = {
            "_id": building_id,
            "building_name": building_name,
            "floors": floors,
            "nodes": merged_nodes,
        }
        return cls.from_mongo_doc(doc)

    # -- Queries --

    def get_node(self, node_id: str) -> Optional[NodeData]:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> list[tuple[str, float, EdgeData]]:
        return self.adjacency.get(node_id, [])

    def get_locations(self) -> list[NodeData]:
        return [n for n in self.nodes.values() if n.type != "junction"]

    def get_all_nodes(self) -> list[NodeData]:
        return list(self.nodes.values())

    def resolve_destination(self, query: str) -> Optional[NodeData]:
        """4-pass fuzzy search: exact name, alias, partial name, partial alias."""
        q = query.lower().strip()

        for node in self.nodes.values():
            if node.type == "junction":
                continue
            if node.name.lower() == q:
                return node

        for node in self.nodes.values():
            if node.type == "junction":
                continue
            for alias in node.aliases:
                if alias.lower() == q:
                    return node

        for node in self.nodes.values():
            if node.type == "junction":
                continue
            if q in node.name.lower():
                return node

        for node in self.nodes.values():
            if node.type == "junction":
                continue
            for alias in node.aliases:
                if q in alias.lower():
                    return node

        return None

    def search_locations(self, query: str, max_results: int = 10) -> list[NodeData]:
        """Multi-pass fuzzy search returning up to max_results non-junction nodes.

        Pass order: exact name, exact alias, partial name, partial alias.
        Earlier passes take priority; a node appears at most once in the result.
        """
        q = query.lower().strip()
        seen: set[str] = set()
        results: list[NodeData] = []

        def _collect(node: NodeData) -> bool:
            if node.id in seen:
                return False
            seen.add(node.id)
            results.append(node)
            return len(results) >= max_results

        for node in self.nodes.values():
            if node.type == "junction":
                continue
            if node.name.lower() == q:
                if _collect(node):
                    return results

        for node in self.nodes.values():
            if node.type == "junction":
                continue
            for alias in node.aliases:
                if alias.lower() == q:
                    if _collect(node):
                        return results
                    break

        for node in self.nodes.values():
            if node.type == "junction":
                continue
            if q in node.name.lower():
                if _collect(node):
                    return results

        for node in self.nodes.values():
            if node.type == "junction":
                continue
            for alias in node.aliases:
                if q in alias.lower():
                    if _collect(node):
                        return results
                    break

        return results

    def euclidean_distance(self, id_a: str, id_b: str) -> float:
        a = self.nodes[id_a]
        b = self.nodes[id_b]
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    # -- Export --

    def to_export_dict(self) -> dict:
        """Serialize back to editor-compatible format."""
        nodes_out = []
        for node in self.nodes.values():
            entry = {
                "id": node.id,
                "type": node.type,
                "floor": node.floor,
                "cx": node.x,
                "cy": node.y,
                "connection": [
                    nid for nid, _, _ in self.adjacency.get(node.id, [])
                ],
            }
            if node.type != "junction":
                entry["objectName"] = node.name
                entry["categoryId"] = node.category
                entry["label"] = node.description
                entry["aliases"] = node.aliases
            nodes_out.append(entry)

        return {
            "_id": self.building_id,
            "building_name": self.building_name,
            "floors": self.floors,
            "nodes": nodes_out,
        }

    # -- Internal --

    @staticmethod
    def _euclidean(a: NodeData, b: NodeData) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)


class GraphRegistry:
    def __init__(self) -> None:
        self.graphs: dict[str, HospitalGraph] = {}

    def get(self, building_id: str) -> Optional[HospitalGraph]:
        return self.graphs.get(building_id)

    def get_default(self) -> Optional[HospitalGraph]:
        if self.graphs:
            return next(iter(self.graphs.values()))
        return None

    def register(self, building_id: str, graph: HospitalGraph) -> None:
        self.graphs[building_id] = graph

    def list_buildings(self) -> list[dict]:
        return [
            {
                "building_id": g.building_id,
                "building_name": g.building_name,
                "floors": g.floors,
                "node_count": g.node_count,
                "edge_count": g.edge_count,
            }
            for g in self.graphs.values()
        ]

    @property
    def building_count(self) -> int:
        return len(self.graphs)
