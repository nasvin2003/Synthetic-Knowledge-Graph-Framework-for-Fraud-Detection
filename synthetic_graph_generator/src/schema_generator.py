import json
from collections import defaultdict

def infer_type(values):
    has_bool = False
    has_int = False
    has_float = False
    has_str = False

    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            has_bool = True
        elif isinstance(v, int):
            has_int = True
        elif isinstance(v, float):
            has_float = True
        elif isinstance(v, str):
            has_str = True
        else:
            has_str = True

    if has_str:
        return "string"
    if has_float:
        return "float"
    if has_int and not has_bool:
        return "integer"
    if has_bool and not has_int and not has_float:
        return "boolean"
    if has_bool and has_int:
        return "integer"
    return "string"


def is_unique_non_null(values):
    seen = set()
    non_null_count = 0
    for v in values:
        if v is None:
            continue
        non_null_count += 1
        if v in seen:
            return False
        seen.add(v)
    return non_null_count > 0


def infer_node_schema(kg):
    nodes_schema = []

    for label, node_ids in kg.nodes_by_label.items():
        prop_values = defaultdict(list)

        for nid in node_ids:
            props = kg.node_props.get(nid, {})
            for k, v in props.items():
                if k == "id":
                    continue
                prop_values[k].append(v)

        properties = []
        for pname, vals in sorted(prop_values.items()):
            ptype = infer_type(vals)

            prop_def = {
                "name": pname,
                "type": ptype
            }

            if ptype == "integer":
                numeric_vals = [v for v in vals if isinstance(v, int) and not isinstance(v, bool)]
                if numeric_vals:
                    prop_def["min"] = min(numeric_vals)
                    prop_def["max"] = max(numeric_vals)

            elif ptype == "float":
                numeric_vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if numeric_vals:
                    prop_def["min"] = min(numeric_vals)
                    prop_def["max"] = max(numeric_vals)

            if is_unique_non_null(vals):
                prop_def["unique"] = True

            properties.append(prop_def)

        nodes_schema.append({
            "label": label,
            "count": len(node_ids),
            "properties": properties
        })

    return nodes_schema


def infer_relationship_schema(kg):
    relationships_schema = []

    for rel_type, edges in kg.edges_by_type.items():
        if not edges:
            continue

        pair_counts = defaultdict(int)
        out_deg = defaultdict(int)
        in_deg = defaultdict(int)
        edge_prop_values = defaultdict(list)

        for e in edges:
            src_label = kg.node_label.get(e.src, "Unknown")
            dst_label = kg.node_label.get(e.dst, "Unknown")

            pair_counts[(src_label, dst_label)] += 1
            out_deg[e.src] += 1
            in_deg[e.dst] += 1

            for k, v in e.props.items():
                edge_prop_values[k].append(v)

        # split one rel_type into separate schema entries if it connects multiple label pairs
        for (from_label, to_label), pair_count in sorted(pair_counts.items()):
            pair_edges = [e for e in edges if kg.node_label.get(e.src) == from_label and kg.node_label.get(e.dst) == to_label]

            pair_out_deg = defaultdict(int)
            pair_in_deg = defaultdict(int)
            allow_self_loop = False

            for e in pair_edges:
                pair_out_deg[e.src] += 1
                pair_in_deg[e.dst] += 1
                if e.src == e.dst:
                    allow_self_loop = True

            from_nodes = kg.nodes_by_label.get(from_label, [])
            to_nodes = kg.nodes_by_label.get(to_label, [])

            observed_out = [pair_out_deg.get(nid, 0) for nid in from_nodes]
            observed_in = [pair_in_deg.get(nid, 0) for nid in to_nodes]

            from_degree = {
                "min": min(observed_out) if observed_out else 0,
                "max": max(observed_out) if observed_out else 0
            }

            to_degree = {
                "min": min(observed_in) if observed_in else 0,
                "max": max(observed_in) if observed_in else 0
            }

            properties = []
            local_edge_prop_values = defaultdict(list)
            for e in pair_edges:
                for k, v in e.props.items():
                    local_edge_prop_values[k].append(v)

            for pname, vals in sorted(local_edge_prop_values.items()):
                ptype = infer_type(vals)
                prop_def = {
                    "name": pname,
                    "type": ptype
                }

                if ptype == "integer":
                    numeric_vals = [v for v in vals if isinstance(v, int) and not isinstance(v, bool)]
                    if numeric_vals:
                        prop_def["min"] = min(numeric_vals)
                        prop_def["max"] = max(numeric_vals)

                elif ptype == "float":
                    numeric_vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                    if numeric_vals:
                        prop_def["min"] = min(numeric_vals)
                        prop_def["max"] = max(numeric_vals)

                properties.append(prop_def)

            relationships_schema.append({
                "type": rel_type,
                "from_node": from_label,
                "to_node": to_label,
                "count": len(pair_edges),
                "constraints": {
                    "allow_self_loop": allow_self_loop,
                    "from_degree": from_degree,
                    "to_degree": to_degree
                },
                "properties": properties
            })

    return relationships_schema


def infer_schema_from_kg(kg, seed=42, no_duplicate_triples=True):
    schema = {
        "seed": seed,
        "no_duplicate_triples": no_duplicate_triples,
        "nodes": infer_node_schema(kg),
        "relationships": infer_relationship_schema(kg)
    }
    return schema


def save_inferred_schema(kg, output_path, seed=42, no_duplicate_triples=True):
    schema = infer_schema_from_kg(
        kg,
        seed=seed,
        no_duplicate_triples=no_duplicate_triples
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema