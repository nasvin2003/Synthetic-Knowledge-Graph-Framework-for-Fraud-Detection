import math
import random
from collections import defaultdict
from typing import Any, Dict, List

from schema_text_parser import load_schema_definition
from utils import clamp, sample_int_uniform, sample_str, powerlaw_weights, cdf_from_weights, sample_from_cdf, save_kg_to_neo4j, KG


def allocate_degrees_powerlaw_exact_sum(
    n: int,
    kmin: int,
    kmax: int,
    total: int,
    alpha: float,
    rng: random.Random,
) -> List[int]:
    if n <= 0:
        return []

    min_total = n * kmin
    max_total = n * kmax
    if total < min_total:
        eff_kmin = total // n
        eff_kmin = clamp(eff_kmin, 0, kmax)
        kmin = eff_kmin
        min_total = n * kmin
    if total > max_total:
        total = max_total

    deg = [kmin] * n
    remaining = total - n * kmin
    if remaining == 0:
        return deg

    cap = [kmax - kmin] * n

    extra_min = 0
    extra_max = kmax - kmin
    weights = powerlaw_weights(extra_min, extra_max, alpha)
    cdf = cdf_from_weights(weights)

    extras = []
    for _ in range(n):
        x = sample_from_cdf(cdf, rng)
        extras.append(x)

    for i in range(n):
        x = extras[i]
        if x > cap[i]:
            x = cap[i]
        deg[i] += x
        remaining -= x

    if remaining < 0:
        need_remove = -remaining
        candidates = [i for i in range(n) if deg[i] > kmin]
        while need_remove > 0 and candidates:
            weights2 = []
            for i in candidates:
                weights2.append(1.0 / (deg[i] - kmin + 1.0))
            s = sum(weights2)
            r = rng.random() * s
            acc = 0.0
            chosen_idx = 0
            for j, w in enumerate(weights2):
                acc += w
                if acc >= r:
                    chosen_idx = j
                    break
            i = candidates[chosen_idx]
            deg[i] -= 1
            need_remove -= 1
            if deg[i] == kmin:
                candidates.pop(chosen_idx)
        remaining = 0
    elif remaining > 0:
        need_add = remaining
        candidates = [i for i in range(n) if deg[i] < kmax]
        while need_add > 0 and candidates:
            weights2 = []
            for i in candidates:
                weights2.append(deg[i] - kmin + 1.0)
            s = sum(weights2)
            r = rng.random() * s
            acc = 0.0
            chosen_idx = 0
            for j, w in enumerate(weights2):
                acc += w
                if acc >= r:
                    chosen_idx = j
                    break
            i = candidates[chosen_idx]
            deg[i] += 1
            need_add -= 1
            if deg[i] == kmax:
                candidates.pop(chosen_idx)
        remaining = 0

    for i in range(n):
        deg[i] = clamp(deg[i], kmin, kmax)

    diff = total - sum(deg)
    if diff != 0:
        if diff > 0:
            candidates = [i for i in range(n) if deg[i] < kmax]
            for _ in range(diff):
                if not candidates:
                    break
                i = rng.choice(candidates)
                deg[i] += 1
                if deg[i] == kmax:
                    candidates.remove(i)
        else:
            candidates = [i for i in range(n) if deg[i] > kmin]
            for _ in range(-diff):
                if not candidates:
                    break
                i = rng.choice(candidates)
                deg[i] -= 1
                if deg[i] == kmin:
                    candidates.remove(i)

    return deg



def _gen_node_properties(node_def: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    props_defs = node_def.get("properties", [])
    count = int(node_def.get("count", 0))

    unique_track: Dict[str, set] = defaultdict(set)

    result: List[Dict[str, Any]] = []
    for _ in range(count):
        props: Dict[str, Any] = {}
        for p in props_defs:
            pname = p["name"]
            ptype = p["type"]
            unique = bool(p.get("unique", False))

            if ptype in ("integer", "int"):
                lo = int(p.get("min", 0))
                hi = int(p.get("max", lo))
                val = sample_int_uniform(lo, hi, rng)

            elif ptype in ("float", "double"):
                lo = float(p.get("min", 0.0))
                hi = float(p.get("max", lo))
                if pname.lower() == "rating":
                    lo_i = int(math.ceil(lo))
                    hi_i = int(math.floor(hi))
                    val = float(rng.randint(lo_i, hi_i))
                else:
                    val = round(rng.uniform(lo, hi), 4)

            elif ptype in ("boolean", "bool"):
                val = bool(rng.getrandbits(1))

            elif ptype in ("string", "str"):
                val = sample_str(rng)

            else:
                val = None

            if unique:
                tries = 0
                while val in unique_track[pname] and tries < 100:
                    if ptype in ("integer", "int"):
                        lo = int(p.get("min", 0))
                        hi = int(p.get("max", lo))
                        val = sample_int_uniform(lo, hi, rng)
                    elif ptype in ("float", "double"):
                        lo = float(p.get("min", 0.0))
                        hi = float(p.get("max", lo))
                        if pname.lower() == "rating":
                            lo_i = int(math.ceil(lo))
                            hi_i = int(math.floor(hi))
                            val = float(rng.randint(lo_i, hi_i))
                        else:
                            val = round(rng.uniform(lo, hi), 4)
                    elif ptype in ("boolean", "bool"):
                        val = bool(rng.getrandbits(1))
                    else:
                        val = sample_str(rng)
                    tries += 1
                unique_track[pname].add(val)

            props[pname] = val

        result.append(props)
    return result



def _gen_edge_properties(rel_def: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for p in rel_def.get("properties", []) or []:
        pname = p["name"]
        ptype = p["type"]

        if ptype in ("integer", "int"):
            lo = int(p.get("min", 0))
            hi = int(p.get("max", lo))
            props[pname] = sample_int_uniform(lo, hi, rng)

        elif ptype in ("float", "double"):
            lo = float(p.get("min", 0.0))
            hi = float(p.get("max", lo))
            props[pname] = round(rng.uniform(lo, hi), 4)

        elif ptype in ("boolean", "bool"):
            props[pname] = bool(rng.getrandbits(1))

        elif ptype in ("string", "str"):
            props[pname] = sample_str(rng)

        else:
            props[pname] = None

    return props



def generate_kg_from_schema(schema_path: str, alpha: float = 2.3) -> KG:
    schema = load_schema_definition(schema_path)

    seed = int(schema.get("seed", 0) or 0)
    rng = random.Random(seed)

    no_dup = bool(schema.get("no_duplicate_triples", True))

    g = KG()

    for node_def in schema.get("nodes", []):
        label = node_def["label"]
        props_list = _gen_node_properties(node_def, rng)
        for props in props_list:
            g.add_node(label, props)

    for rel_def in schema.get("relationships", []):
        rel_type = rel_def["type"]
        from_label = rel_def["from_node"]
        to_label = rel_def["to_node"]
        count = int(rel_def.get("count", 0))

        constraints = rel_def.get("constraints", {}) or {}
        allow_self = bool(constraints.get("allow_self_loop", True))

        sources = g.nodes_by_label.get(from_label, [])
        targets = g.nodes_by_label.get(to_label, [])

        if count <= 0 or not sources or not targets:
            continue

        fd = constraints.get("from_degree", {}) or {}
        td = constraints.get("to_degree", {}) or {}

        out_min = int(fd.get("min", 0))
        out_max = int(fd.get("max", max(1, len(targets) - (0 if allow_self else 1))))

        in_min = int(td.get("min", 0))
        in_max = int(td.get("max", max(1, len(sources) - (0 if allow_self else 1))))

        out_max = max(out_max, out_min)
        in_max = max(in_max, in_min)

        if not allow_self and from_label == to_label and len(targets) > 0:
            out_max = min(out_max, len(targets) - 1)
            in_max = min(in_max, len(sources) - 1)

        out_deg = allocate_degrees_powerlaw_exact_sum(len(sources), out_min, out_max, count, alpha, rng)
        in_deg = allocate_degrees_powerlaw_exact_sum(len(targets), in_min, in_max, count, alpha, rng)

        out_stubs: List[int] = []
        for sid, d in zip(sources, out_deg):
            if d > 0:
                out_stubs.extend([sid] * d)

        in_stubs: List[int] = []
        for tid, d in zip(targets, in_deg):
            if d > 0:
                in_stubs.extend([tid] * d)

        out_stubs = out_stubs[:count]
        in_stubs = in_stubs[:count]

        rng.shuffle(out_stubs)

        edges_made = 0
        max_global_tries = max(50_000, count * 10)
        tries = 0

        while edges_made < count and tries < max_global_tries and out_stubs:
            s = out_stubs.pop()
            placed = False
            local_tries = 0
            while not placed and local_tries < 30 and in_stubs:
                idx = rng.randrange(len(in_stubs))
                t = in_stubs[idx]

                if (not allow_self) and s == t:
                    local_tries += 1
                    tries += 1
                    continue

                key = (s << 32) | t
                if no_dup and key in g.edge_keys[rel_type]:
                    local_tries += 1
                    tries += 1
                    continue

                in_stubs[idx] = in_stubs[-1]
                in_stubs.pop()

                props = _gen_edge_properties(rel_def, rng)
                g.add_edge(rel_type, s, t, props, no_duplicates=no_dup)
                edges_made += 1
                placed = True

            if not placed:
                tries += 1

    return g



def main() -> None:
    schema = "synthetic_graph_generator/schemas/amazon_inferred_schema.txt"
    kg = generate_kg_from_schema(schema)
    print("Total nodes:", len(kg.nodes))
    print("Total edges:", sum(len(v) for v in kg.edges_by_type.values()))

    save_kg_to_neo4j(kg, uri="bolt://localhost:7687", user="neo4j", password="password", database="neo4j")


if __name__ == "__main__":
    main()
