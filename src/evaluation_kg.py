import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from kg_generator import generate_kg_from_schema
from utils import KG, build_amazon_reviews_kg

try:
    from scipy.stats import chi2 as scipy_chi2_dist
    SCIPY_AVAILABLE = True
except Exception:
    scipy_chi2_dist = None
    SCIPY_AVAILABLE = False



def total_edge_count(kg: KG) -> int:
    return sum(len(edges) for edges in kg.edges_by_type.values())


def node_counts_by_label(kg: KG) -> Dict[str, int]:
    return {label: len(ids) for label, ids in kg.nodes_by_label.items()}


def edge_counts_by_type(kg: KG) -> Dict[str, int]:
    return {rel_type: len(edges) for rel_type, edges in kg.edges_by_type.items()}


def safe_relative_error(original: float, generated: float) -> Optional[float]:
    if original == 0:
        return 0.0 if generated == 0 else None
    return abs(generated - original) / abs(original)


def summary_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    n = len(values)
    if n == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
    }


def discrete_hist(values: Sequence[Any]) -> Dict[str, int]:
    hist = Counter(values)
    return {str(k): int(v) for k, v in sorted(hist.items(), key=lambda x: x[0])}


def numeric_hist(values: Sequence[float], bins: int = 20) -> Dict[str, int]:
    if not values:
        return {}

    unique_vals = sorted(set(values))
    if len(unique_vals) <= 50:
        return discrete_hist(values)

    lo = min(values)
    hi = max(values)
    if lo == hi:
        return {f"[{lo}, {hi}]": len(values)}

    width = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        idx = int((value - lo) / width)
        if idx == bins:
            idx -= 1
        counts[idx] += 1

    hist: Dict[str, int] = {}
    for i, count in enumerate(counts):
        left = lo + i * width
        right = lo + (i + 1) * width
        hist[f"[{left:.4f}, {right:.4f})"] = count
    return hist


def normalize_counter(counter: Dict[str, int], smoothing: float = 0.0) -> Dict[str, float]:
    keys = list(counter.keys())
    total = sum(counter.values()) + smoothing * len(keys)
    if total == 0:
        return {k: 0.0 for k in keys}
    return {k: (counter.get(k, 0) + smoothing) / total for k in keys}


def js_divergence(counts_a: Dict[str, int], counts_b: Dict[str, int], smoothing: float = 1e-12) -> Optional[float]:
    keys = sorted(set(counts_a.keys()) | set(counts_b.keys()))
    if not keys:
        return None

    pa_raw = {k: counts_a.get(k, 0) for k in keys}
    pb_raw = {k: counts_b.get(k, 0) for k in keys}

    pa = normalize_counter(pa_raw, smoothing=smoothing)
    pb = normalize_counter(pb_raw, smoothing=smoothing)
    m = {k: 0.5 * (pa[k] + pb[k]) for k in keys}

    def kl_div(p: Dict[str, float], q: Dict[str, float]) -> float:
        value = 0.0
        for key in keys:
            if p[key] > 0 and q[key] > 0:
                value += p[key] * math.log(p[key] / q[key], 2)
        return value

    return 0.5 * kl_div(pa, m) + 0.5 * kl_div(pb, m)


def chi_square_gof(
    original_hist: Dict[str, int],
    generated_hist: Dict[str, int],
    laplace_smoothing: float = 1.0,
) -> Dict[str, Optional[float]]:
    keys = sorted(set(original_hist.keys()) | set(generated_hist.keys()))
    if not keys:
        return {"chi2": None, "p_value": None, "degrees_of_freedom": None}

    original_total = sum(original_hist.get(k, 0) for k in keys)
    generated_total = sum(generated_hist.get(k, 0) for k in keys)
    if original_total == 0 or generated_total == 0:
        return {"chi2": None, "p_value": None, "degrees_of_freedom": None}

    smoothed_den = original_total + laplace_smoothing * len(keys)
    expected = [((original_hist.get(k, 0) + laplace_smoothing) / smoothed_den) * generated_total for k in keys]
    observed = [generated_hist.get(k, 0) for k in keys]

    chi2_stat = 0.0
    for obs, exp in zip(observed, expected):
        if exp > 0:
            chi2_stat += ((obs - exp) ** 2) / exp

    dof = max(1, len(keys) - 1)
    p_value = float(scipy_chi2_dist.sf(chi2_stat, dof)) if SCIPY_AVAILABLE else None

    return {
        "chi2": chi2_stat,
        "p_value": p_value,
        "degrees_of_freedom": dof,
    }


def infer_relation_signatures(kg: KG) -> Dict[str, set]:
    signatures: Dict[str, set] = defaultdict(set)
    for rel_type, edges in kg.edges_by_type.items():
        for edge in edges:
            src_label = kg.node_label.get(edge.src)
            dst_label = kg.node_label.get(edge.dst)
            signatures[rel_type].add((src_label, dst_label))
    return signatures


def degree_values_for_signature(
    kg: KG,
    rel_type: str,
    src_label: str,
    dst_label: str,
) -> Tuple[List[int], List[int]]:
    src_nodes = kg.nodes_by_label.get(src_label, [])
    dst_nodes = kg.nodes_by_label.get(dst_label, [])

    out_deg = {nid: 0 for nid in src_nodes}
    in_deg = {nid: 0 for nid in dst_nodes}

    for edge in kg.edges_by_type.get(rel_type, []):
        edge_src_label = kg.node_label.get(edge.src)
        edge_dst_label = kg.node_label.get(edge.dst)
        if edge_src_label == src_label and edge_dst_label == dst_label:
            if edge.src in out_deg:
                out_deg[edge.src] += 1
            if edge.dst in in_deg:
                in_deg[edge.dst] += 1

    return list(out_deg.values()), list(in_deg.values())


def compare_count_maps(
    original_counts: Dict[str, int],
    generated_counts: Dict[str, int],
) -> Dict[str, Dict[str, Optional[float]]]:
    result: Dict[str, Dict[str, Optional[float]]] = {}
    keys = sorted(set(original_counts.keys()) | set(generated_counts.keys()))
    for key in keys:
        orig = int(original_counts.get(key, 0))
        gen = int(generated_counts.get(key, 0))
        result[key] = {
            "original": orig,
            "generated": gen,
            "absolute_difference": abs(gen - orig),
            "relative_error": safe_relative_error(orig, gen),
        }
    return result


def compare_degree_distributions(original_kg: KG, generated_kg: KG) -> Dict[str, Any]:
    sig_orig = infer_relation_signatures(original_kg)
    sig_gen = infer_relation_signatures(generated_kg)

    rel_types = sorted(set(sig_orig.keys()) | set(sig_gen.keys()))
    report: Dict[str, Any] = {}

    for rel_type in rel_types:
        all_signatures = sorted(sig_orig.get(rel_type, set()) | sig_gen.get(rel_type, set()))
        for src_label, dst_label in all_signatures:
            key = f"{rel_type}|{src_label}|{dst_label}"
            orig_out, orig_in = degree_values_for_signature(original_kg, rel_type, src_label, dst_label)
            gen_out, gen_in = degree_values_for_signature(generated_kg, rel_type, src_label, dst_label)

            orig_out_hist = discrete_hist(orig_out)
            gen_out_hist = discrete_hist(gen_out)
            orig_in_hist = discrete_hist(orig_in)
            gen_in_hist = discrete_hist(gen_in)

            report[key] = {
                "relation_type": rel_type,
                "source_label": src_label,
                "target_label": dst_label,
                "source_degree": {
                    "original_stats": summary_stats(orig_out),
                    "generated_stats": summary_stats(gen_out),
                    "mean_relative_error": safe_relative_error(
                        summary_stats(orig_out)["mean"] or 0.0,
                        summary_stats(gen_out)["mean"] or 0.0,
                    ),
                    "max_relative_error": safe_relative_error(
                        summary_stats(orig_out)["max"] or 0.0,
                        summary_stats(gen_out)["max"] or 0.0,
                    ),
                    "js_divergence": js_divergence(orig_out_hist, gen_out_hist),
                    "chi_square": chi_square_gof(orig_out_hist, gen_out_hist),
                    "original_histogram": orig_out_hist,
                    "generated_histogram": gen_out_hist,
                },
                "target_degree": {
                    "original_stats": summary_stats(orig_in),
                    "generated_stats": summary_stats(gen_in),
                    "mean_relative_error": safe_relative_error(
                        summary_stats(orig_in)["mean"] or 0.0,
                        summary_stats(gen_in)["mean"] or 0.0,
                    ),
                    "max_relative_error": safe_relative_error(
                        summary_stats(orig_in)["max"] or 0.0,
                        summary_stats(gen_in)["max"] or 0.0,
                    ),
                    "js_divergence": js_divergence(orig_in_hist, gen_in_hist),
                    "chi_square": chi_square_gof(orig_in_hist, gen_in_hist),
                    "original_histogram": orig_in_hist,
                    "generated_histogram": gen_in_hist,
                },
            }
    return report


def build_evaluation_report(original_kg: KG, generated_kg: KG, category: str, schema_path: str) -> Dict[str, Any]:
    original_node_counts = node_counts_by_label(original_kg)
    generated_node_counts = node_counts_by_label(generated_kg)
    original_edge_counts = edge_counts_by_type(original_kg)
    generated_edge_counts = edge_counts_by_type(generated_kg)

    report = {
        "metadata": {
            "category": category,
            "schema_path": schema_path,
            "scipy_available": SCIPY_AVAILABLE,
            "chi_square_note": "Chi-square uses Laplace smoothing on the original histogram so zero-probability bins do not create infinite statistics.",
        },
        "global": {
            "original_total_nodes": len(original_kg.nodes),
            "generated_total_nodes": len(generated_kg.nodes),
            "node_count_relative_error": safe_relative_error(len(original_kg.nodes), len(generated_kg.nodes)),
            "original_total_edges": total_edge_count(original_kg),
            "generated_total_edges": total_edge_count(generated_kg),
            "edge_count_relative_error": safe_relative_error(total_edge_count(original_kg), total_edge_count(generated_kg)),
        },
        "node_counts_by_label": compare_count_maps(original_node_counts, generated_node_counts),
        "edge_counts_by_type": compare_count_maps(original_edge_counts, generated_edge_counts),
        "degree_distributions": compare_degree_distributions(original_kg, generated_kg),
    }
    return report


def print_summary(report: Dict[str, Any]) -> None:
    print("\n=== GLOBAL ===")
    global_stats = report["global"]
    print(f"Original total nodes:   {global_stats['original_total_nodes']}")
    print(f"Generated total nodes:  {global_stats['generated_total_nodes']}")
    print(f"Node count rel. error:  {global_stats['node_count_relative_error']}")
    print(f"Original total edges:   {global_stats['original_total_edges']}")
    print(f"Generated total edges:  {global_stats['generated_total_edges']}")
    print(f"Edge count rel. error:  {global_stats['edge_count_relative_error']}")

    print("\n=== NODE COUNTS BY LABEL ===")
    for label, stats in report["node_counts_by_label"].items():
        print(
            f"{label}: original={stats['original']}, generated={stats['generated']}, "
            f"abs_diff={stats['absolute_difference']}, rel_error={stats['relative_error']}"
        )

    print("\n=== EDGE COUNTS BY TYPE ===")
    for rel_type, stats in report["edge_counts_by_type"].items():
        print(
            f"{rel_type}: original={stats['original']}, generated={stats['generated']}, "
            f"abs_diff={stats['absolute_difference']}, rel_error={stats['relative_error']}"
        )

    print("\n=== DEGREE DISTRIBUTIONS ===")
    for key, block in report["degree_distributions"].items():
        src = block["source_degree"]
        dst = block["target_degree"]
        print(key)
        print(
            "  source: "
            f"mean(orig)={src['original_stats']['mean']}, "
            f"mean(gen)={src['generated_stats']['mean']}, "
            f"js={src['js_divergence']}, "
            f"chi2={src['chi_square']['chi2']}, "
            f"p={src['chi_square']['p_value']}"
        )
        print(
            "  target: "
            f"mean(orig)={dst['original_stats']['mean']}, "
            f"mean(gen)={dst['generated_stats']['mean']}, "
            f"js={dst['js_divergence']}, "
            f"chi2={dst['chi_square']['chi2']}, "
            f"p={dst['chi_square']['p_value']}"
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a schema-generated KG against the original Amazon reviews KG.")
    parser.add_argument(
        "--category",
        type=str,
        default="raw_review_Digital_Music",
        help="Amazon Reviews 2023 category used to build the original KG.",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="schemas/json/small_schema.json",
        help="Path to the inferred schema JSON file used to generate the synthetic KG.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/evaluation_kg_report.json",
        help="Path to save the evaluation report as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Building original KG from Amazon dataset...")
    original_kg = build_amazon_reviews_kg(args.category)

    print("Generating synthetic KG from schema...")
    generated_kg = generate_kg_from_schema(args.schema)

    print("Computing evaluation metrics...")
    report = build_evaluation_report(original_kg, generated_kg, args.category, args.schema)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"\nSaved report to: {args.output}")


if __name__ == "__main__":
    main()
