from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List

from schema_text_parser import load_schema_definition
from utils import (
    KG,
    cdf_from_weights,
    clamp,
    powerlaw_weights,
    sample_from_cdf,
    sample_int_uniform,
    sample_str,
)


def _deterministic_unique_value(
    label: str,
    pname: str,
    ptype: str,
    idx: int,
    spec: Dict[str, Any],
) -> Any:
    if ptype in ("string", "str"):
        return f"{label.lower()}_{pname.lower()}_{idx}"

    if ptype in ("integer", "int"):
        lo = int(spec.get("min", 0))
        hi = int(spec.get("max", lo + idx))
        span = hi - lo + 1
        if span <= 0:
            return lo + idx
        if idx < span:
            return lo + idx
        return lo + (idx % span)

    if ptype in ("float", "double"):
        lo = float(spec.get("min", 0.0))
        hi = float(spec.get("max", lo + idx))
        if hi <= lo:
            return lo + float(idx)
        return lo + float(idx % max(1, int(hi - lo + 1)))

    if ptype in ("boolean", "bool"):
        return bool(idx & 1)

    return f"{label.lower()}_{pname.lower()}_{idx}"


def _iter_node_properties(node_def: Dict[str, Any], rng: random.Random) -> Iterable[Dict[str, Any]]:
    props_defs = node_def.get("properties", [])
    count = int(node_def.get("count", 0))
    label = str(node_def.get("label", "Node"))

    for idx in range(count):
        props: Dict[str, Any] = {}
        for p in props_defs:
            pname = p["name"]
            ptype = p["type"]
            unique = bool(p.get("unique", False))

            if unique:
                props[pname] = _deterministic_unique_value(label, pname, ptype, idx, p)
                continue

            if ptype in ("integer", "int"):
                lo = int(p.get("min", 0))
                hi = int(p.get("max", lo))
                props[pname] = sample_int_uniform(lo, hi, rng)
            elif ptype in ("float", "double"):
                lo = float(p.get("min", 0.0))
                hi = float(p.get("max", lo))
                if pname.lower() == "rating":
                    lo_i = int(math.ceil(lo))
                    hi_i = int(math.floor(hi))
                    props[pname] = float(rng.randint(lo_i, hi_i))
                else:
                    props[pname] = round(rng.uniform(lo, hi), 4)
            elif ptype in ("boolean", "bool"):
                props[pname] = bool(rng.getrandbits(1))
            elif ptype in ("string", "str"):
                props[pname] = sample_str(rng)
            else:
                props[pname] = None

        yield props


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


def _bulk_adjust_degrees(
    deg: List[int],
    kmin: int,
    kmax: int,
    target_total: int,
    rng: random.Random,
) -> None:
    cur = sum(deg)
    diff = target_total - cur
    if diff == 0:
        return

    if diff > 0:
        candidates = [i for i, d in enumerate(deg) if d < kmax]
        if not candidates:
            return
        rng.shuffle(candidates)
        n = len(candidates)
        ptr = 0
        while diff > 0 and n > 0:
            i = candidates[ptr]
            room = kmax - deg[i]
            if room > 0:
                add = min(room, max(1, diff // max(1, n)))
                deg[i] += add
                diff -= add
            ptr += 1
            if ptr >= n:
                ptr = 0
                candidates = [j for j in candidates if deg[j] < kmax]
                n = len(candidates)
                if n == 0:
                    break
                ptr %= n
    else:
        need = -diff
        candidates = [i for i, d in enumerate(deg) if d > kmin]
        if not candidates:
            return
        rng.shuffle(candidates)
        n = len(candidates)
        ptr = 0
        while need > 0 and n > 0:
            i = candidates[ptr]
            room = deg[i] - kmin
            if room > 0:
                rem = min(room, max(1, need // max(1, n)))
                deg[i] -= rem
                need -= rem
            ptr += 1
            if ptr >= n:
                ptr = 0
                candidates = [j for j in candidates if deg[j] > kmin]
                n = len(candidates)
                if n == 0:
                    break
                ptr %= n


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
    remaining = total - min_total
    if remaining <= 0:
        return deg

    extra_max = kmax - kmin
    if extra_max <= 0:
        return deg

    weights = powerlaw_weights(0, extra_max, alpha)
    cdf = cdf_from_weights(weights)

    sampled_extra = [0] * n
    sampled_sum = 0
    for i in range(n):
        x = sample_from_cdf(cdf, rng)
        if x > extra_max:
            x = extra_max
        sampled_extra[i] = x
        sampled_sum += x

    if sampled_sum == 0:
        sampled_sum = 1

    running_sum = 0
    for i, x in enumerate(sampled_extra):
        scaled = int((x * remaining) / sampled_sum)
        if scaled > extra_max:
            scaled = extra_max
        deg[i] += scaled
        running_sum += scaled

    _bulk_adjust_degrees(deg, kmin, kmax, total, rng)
    return deg


def _relation_constraints(
    rel_def: Dict[str, Any],
    num_sources: int,
    num_targets: int,
) -> tuple[bool, int, int, int, int]:
    constraints = rel_def.get("constraints", {}) or {}
    allow_self = bool(constraints.get("allow_self_loop", True))

    fd = constraints.get("from_degree", {}) or {}
    td = constraints.get("to_degree", {}) or {}

    out_min = int(fd.get("min", 0))
    out_max = int(fd.get("max", max(1, num_targets - (0 if allow_self else 1))))

    in_min = int(td.get("min", 0))
    in_max = int(td.get("max", max(1, num_sources - (0 if allow_self else 1))))

    out_max = max(out_max, out_min)
    in_max = max(in_max, in_min)
    return allow_self, out_min, out_max, in_min, in_max


def _is_partitionable_relation(
    rel_def: Dict[str, Any],
    from_label: str,
    to_label: str,
    allow_self: bool,
    out_min: int,
    out_max: int,
    in_min: int,
    in_max: int,
) -> bool:
    if from_label == to_label and not allow_self:
        return False

    return (out_min == 1 and out_max == 1) or (in_min == 1 and in_max == 1)


def _generate_partition_relation(
    g: KG,
    rel_def: Dict[str, Any],
    rel_type: str,
    sources: List[int],
    targets: List[int],
    count: int,
    out_min: int,
    out_max: int,
    in_min: int,
    in_max: int,
    alpha: float,
    rng: random.Random,
) -> int:
    edges_made = 0

    if out_min == 1 and out_max == 1:
        if count > len(sources):
            raise ValueError(
                f"Relation {rel_type}: count={count} exceeds number of source nodes {len(sources)} for exact out-degree 1."
            )

        src_ids = list(sources)
        rng.shuffle(src_ids)
        src_ids = src_ids[:count]

        target_deg = allocate_degrees_powerlaw_exact_sum(
            len(targets),
            in_min,
            in_max,
            count,
            alpha,
            rng,
        )

        pos = 0
        for tid, d in zip(targets, target_deg):
            end = pos + d
            for sid in src_ids[pos:end]:
                props = _gen_edge_properties(rel_def, rng)
                g.add_edge(rel_type, sid, tid, props, no_duplicates=False)
                edges_made += 1
            pos = end
        return edges_made

    if in_min == 1 and in_max == 1:
        if count > len(targets):
            raise ValueError(
                f"Relation {rel_type}: count={count} exceeds number of target nodes {len(targets)} for exact in-degree 1."
            )

        tgt_ids = list(targets)
        rng.shuffle(tgt_ids)
        tgt_ids = tgt_ids[:count]

        source_deg = allocate_degrees_powerlaw_exact_sum(
            len(sources),
            out_min,
            out_max,
            count,
            alpha,
            rng,
        )

        pos = 0
        for sid, d in zip(sources, source_deg):
            end = pos + d
            for tid in tgt_ids[pos:end]:
                props = _gen_edge_properties(rel_def, rng)
                g.add_edge(rel_type, sid, tid, props, no_duplicates=False)
                edges_made += 1
            pos = end
        return edges_made

    return edges_made


def _generate_general_relation(
    g: KG,
    rel_def: Dict[str, Any],
    rel_type: str,
    sources: List[int],
    targets: List[int],
    count: int,
    allow_self: bool,
    out_min: int,
    out_max: int,
    in_min: int,
    in_max: int,
    alpha: float,
    rng: random.Random,
    no_dup: bool,
) -> int:
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

    return edges_made


def generate_kg_from_schema(schema_path: str, alpha: float = 2.3, verbose: bool = False) -> KG:
    schema = load_schema_definition(schema_path)

    seed = int(schema.get("seed", 0) or 0)
    rng = random.Random(seed)
    no_dup = bool(schema.get("no_duplicate_triples", True))

    g = KG()

    for node_def in schema.get("nodes", []):
        label = node_def["label"]
        count = int(node_def.get("count", 0))
        if verbose:
            print(f"[optimized] generating {count} nodes for {label}...")
        for props in _iter_node_properties(node_def, rng):
            g.add_node(label, props)

    for rel_def in schema.get("relationships", []):
        rel_type = rel_def["type"]
        from_label = rel_def["from_node"]
        to_label = rel_def["to_node"]
        count = int(rel_def.get("count", 0))

        sources = g.nodes_by_label.get(from_label, [])
        targets = g.nodes_by_label.get(to_label, [])
        if count <= 0 or not sources or not targets:
            continue

        allow_self, out_min, out_max, in_min, in_max = _relation_constraints(
            rel_def,
            len(sources),
            len(targets),
        )

        if not allow_self and from_label == to_label and len(targets) > 0:
            out_max = min(out_max, len(targets) - 1)
            in_max = min(in_max, len(sources) - 1)

        if verbose:
            print(
                f"[optimized] generating relation {rel_type} ({from_label} -> {to_label}), count={count}..."
            )

        if _is_partitionable_relation(
            rel_def,
            from_label,
            to_label,
            allow_self,
            out_min,
            out_max,
            in_min,
            in_max,
        ):
            made = _generate_partition_relation(
                g,
                rel_def,
                rel_type,
                list(sources),
                list(targets),
                count,
                out_min,
                out_max,
                in_min,
                in_max,
                alpha,
                rng,
            )
        else:
            made = _generate_general_relation(
                g,
                rel_def,
                rel_type,
                list(sources),
                list(targets),
                count,
                allow_self,
                out_min,
                out_max,
                in_min,
                in_max,
                alpha,
                rng,
                no_dup,
            )

        if verbose:
            print(f"[optimized] {rel_type}: created {made} edges")

    return g

import time 

def main() -> None:
    curr_time = int(time.time())
    # schema = "synthetic_graph_generator/schemas/amazon_inferred_schema.txt"
    schema = "report_pipeline_outputs/medium/medium_schema.json"

    kg = generate_kg_from_schema(schema)
    print("Total nodes:", len(kg.nodes))
    print("Total edges:", sum(len(v) for v in kg.edges_by_type.values()))
    print("Time taken: {:.2f} seconds".format(time.time() - curr_time))

    # save_kg_to_neo4j(kg, uri="bolt://localhost:7687", user="neo4j", password="password", database="neo4j")


if __name__ == "__main__":
    main()
