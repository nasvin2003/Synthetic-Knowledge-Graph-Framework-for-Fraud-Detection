import json
from collections import defaultdict
from typing import Any, Dict


class PropertyAccumulator:
    def __init__(self, name: str):
        self.name = name
        self.has_bool = False
        self.has_int = False
        self.has_float = False
        self.has_str = False
        self.min_val = None
        self.max_val = None
        self.non_null_count = 0
        self.unique_possible = True
        self.seen = set()

    def _id_like_name(self) -> bool:
        lname = self.name.lower()
        return (
            lname == "asin"
            or lname.endswith("id")
            or lname.endswith("_id")
            or "asin" in lname
        )

    def add(self, value: Any) -> None:
        if value is None:
            return

        self.non_null_count += 1

        if isinstance(value, bool):
            self.has_bool = True
        elif isinstance(value, int):
            self.has_int = True
        elif isinstance(value, float):
            self.has_float = True
        elif isinstance(value, str):
            self.has_str = True
        else:
            self.has_str = True

        if isinstance(value, int) and not isinstance(value, bool):
            self.min_val = value if self.min_val is None else min(self.min_val, value)
            self.max_val = value if self.max_val is None else max(self.max_val, value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            self.min_val = value if self.min_val is None else min(self.min_val, value)
            self.max_val = value if self.max_val is None else max(self.max_val, value)

        if not self.unique_possible:
            return

        if isinstance(value, str) and not self._id_like_name() and len(self.seen) >= 100_000:
            self.unique_possible = False
            self.seen.clear()
            return

        if value in self.seen:
            self.unique_possible = False
            self.seen.clear()
        else:
            self.seen.add(value)

    def inferred_type(self) -> str:
        if self.has_str:
            return "string"
        if self.has_float:
            return "float"
        if self.has_int and not self.has_bool:
            return "integer"
        if self.has_bool and not self.has_int and not self.has_float:
            return "boolean"
        if self.has_bool and self.has_int:
            return "integer"
        return "string"

    def to_schema(self) -> Dict[str, Any]:
        ptype = self.inferred_type()
        prop_def: Dict[str, Any] = {
            "name": self.name,
            "type": ptype,
        }

        if ptype in ("integer", "float") and self.min_val is not None:
            prop_def["min"] = self.min_val
            prop_def["max"] = self.max_val

        if self.non_null_count > 0 and self.unique_possible:
            prop_def["unique"] = True

        return prop_def


def infer_node_schema(kg):
    nodes_schema = []

    for label, node_ids in kg.nodes_by_label.items():
        prop_stats: Dict[str, PropertyAccumulator] = {}

        for nid in node_ids:
            props = kg.node_props.get(nid, {})
            for k, v in props.items():
                if k == "id":
                    continue
                if k not in prop_stats:
                    prop_stats[k] = PropertyAccumulator(k)
                prop_stats[k].add(v)

        properties = [prop_stats[pname].to_schema() for pname in sorted(prop_stats)]

        nodes_schema.append({
            "label": label,
            "count": len(node_ids),
            "properties": properties,
        })

    return nodes_schema


def infer_relationship_schema(kg):
    relationships_schema = []

    for rel_type, edges in kg.edges_by_type.items():
        if not edges:
            continue

        pair_data = {}

        for e in edges:
            from_label = kg.node_label.get(e.src, "Unknown")
            to_label = kg.node_label.get(e.dst, "Unknown")
            pair = (from_label, to_label)

            if pair not in pair_data:
                pair_data[pair] = {
                    "count": 0,
                    "allow_self_loop": False,
                    "out_deg": defaultdict(int),
                    "in_deg": defaultdict(int),
                    "prop_stats": {},
                }

            data = pair_data[pair]
            data["count"] += 1
            data["out_deg"][e.src] += 1
            data["in_deg"][e.dst] += 1
            if e.src == e.dst:
                data["allow_self_loop"] = True

            for k, v in e.props.items():
                if k not in data["prop_stats"]:
                    data["prop_stats"][k] = PropertyAccumulator(k)
                data["prop_stats"][k].add(v)

        for (from_label, to_label), data in sorted(pair_data.items()):
            from_nodes = kg.nodes_by_label.get(from_label, [])
            to_nodes = kg.nodes_by_label.get(to_label, [])

            out_deg = data["out_deg"]
            in_deg = data["in_deg"]

            from_count = len(from_nodes)
            to_count = len(to_nodes)

            from_degree = {
                "min": 0 if len(out_deg) < from_count else min(out_deg.values(), default=0),
                "max": max(out_deg.values(), default=0),
            }
            to_degree = {
                "min": 0 if len(in_deg) < to_count else min(in_deg.values(), default=0),
                "max": max(in_deg.values(), default=0),
            }

            properties = [data["prop_stats"][pname].to_schema() for pname in sorted(data["prop_stats"])]

            relationships_schema.append({
                "type": rel_type,
                "from_node": from_label,
                "to_node": to_label,
                "count": data["count"],
                "constraints": {
                    "allow_self_loop": data["allow_self_loop"],
                    "from_degree": from_degree,
                    "to_degree": to_degree,
                },
                "properties": properties,
            })

    return relationships_schema


def infer_schema_from_kg(kg, seed=42, no_duplicate_triples=True):
    return {
        "seed": seed,
        "no_duplicate_triples": no_duplicate_triples,
        "nodes": infer_node_schema(kg),
        "relationships": infer_relationship_schema(kg),
    }


def save_inferred_schema(kg, output_path, seed=42, no_duplicate_triples=True):
    schema = infer_schema_from_kg(
        kg,
        seed=seed,
        no_duplicate_triples=no_duplicate_triples,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema
