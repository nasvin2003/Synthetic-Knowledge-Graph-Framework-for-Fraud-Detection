from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path
import time 
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import KG, build_amazon_reviews_kg
from src.schema_generator import infer_schema_from_kg
from src.kg_generator import generate_kg_from_schema
from evaluation_kg import build_evaluation_report
from src.review_pattern_sanitizer import SanitizeConfig, ReviewKGPatternSanitizer
from src.review_fraud_injector import FraudInjectionConfig, ReviewKGUserFraudInjector
from gnn import (
    TaskConfig,
    SplitConfig,
    HeteroGenericSAGE,
    build_heterodata_generic,
    make_single_split_masks,
    assign_masks,
    validate_masks,
    run_one_training,
    _binary_scores_from_probs,
    _binary_auc_from_probs,
    set_seed,
)

SIZE_TO_CATEGORY = {
    "small": "raw_review_Digital_Music",
    # "medium": "raw_review_Movies_and_TV",
    "medium": "raw_review_CDs_and_Vinyl",
    "large": "raw_review_Movies_and_TV",
    # "large": "raw_review_Clothing_Shoes_and_Jewelry",
}


def default_sanitize_cfg(seed: int) -> SanitizeConfig:
    return SanitizeConfig(
        seed=seed,
        min_reviews_repeated_star=6,
        dominant_star_ratio_threshold=0.85,
        min_reviews_deviation=4,
        avg_abs_deviation_threshold=1.25,
        min_reviews_group_concentration=6,
        max_group_concentration_threshold=0.80,
        min_same_product_block_size=3,
        same_product_block_share_threshold=0.60,
        min_common_products_overlap=3,
        min_jaccard_overlap=0.60,
        min_overlap_component_size=3,
        min_user_score_to_remove=2,
        max_removed_user_fraction=0.05,
    )


def default_fraud_cfg(seed: int, corruption_rate: float) -> FraudInjectionConfig:
    return FraudInjectionConfig(
        seed=seed,
        corruption_rate=corruption_rate,
        criminal_user_fraction=0.01,
        min_criminal_users=200,
        camouflage_rate=0.30,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_pickle(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def schema_to_pseudo_text(schema: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"@seed={schema.get('seed', 0)}")
    lines.append(f"@no_duplicate_triples={str(bool(schema.get('no_duplicate_triples', True))).lower()}")
    lines.append("")

    for node in schema.get("nodes", []):
        prop_parts = []
        for p in node.get("properties", []):
            type_name = p["type"]
            if type_name == "integer":
                type_name = "int"
            elif type_name == "boolean":
                type_name = "bool"
            opts = []
            if "min" in p:
                opts.append(f"min={p['min']}")
            if "max" in p:
                opts.append(f"max={p['max']}")
            if p.get("unique", False):
                opts.append("unique=true")
            opt_text = f"[{','.join(opts)}]" if opts else ""
            prop_parts.append(f"{p['name']}:{type_name}{opt_text}")
        props_text = ", ".join(prop_parts)
        lines.append(f"{node['label']}({props_text}) [count={node['count']}]")

    if schema.get("relationships"):
        lines.append("")

    for rel in schema.get("relationships", []):
        c = rel.get("constraints", {}) or {}
        opts = [f"count={rel['count']}"]
        opts.append(f"allow_self_loop={str(bool(c.get('allow_self_loop', True))).lower()}")
        fd = c.get("from_degree", {}) or {}
        td = c.get("to_degree", {}) or {}
        if "min" in fd:
            opts.append(f"from_degree.min={fd['min']}")
        if "max" in fd:
            opts.append(f"from_degree.max={fd['max']}")
        if "min" in td:
            opts.append(f"to_degree.min={td['min']}")
        if "max" in td:
            opts.append(f"to_degree.max={td['max']}")
        lines.append(f"{rel['type']}({rel['from_node']} -> {rel['to_node']})")
        lines.append(f"[{', '.join(opts)}]")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def summarize_kg(kg: KG) -> Dict[str, Any]:
    return {
        "total_nodes": len(kg.nodes),
        "total_edges": sum(len(v) for v in kg.edges_by_type.values()),
        "node_counts_by_label": {k: len(v) for k, v in kg.nodes_by_label.items()},
        "edge_counts_by_type": {k: len(v) for k, v in kg.edges_by_type.items()},
    }


def maybe_cache_build_original_kg(category: str, cache_path: Path | None) -> KG:
    if cache_path is not None and cache_path.exists():
        return load_pickle(cache_path)
    kg = build_amazon_reviews_kg(category)
    if cache_path is not None:
        save_pickle(kg, cache_path)
    return kg


def maybe_cache_generate_synthetic(schema_path: Path, cache_path: Path | None) -> KG:
    if cache_path is not None and cache_path.exists():
        return load_pickle(cache_path)
    kg = generate_kg_from_schema(str(schema_path))
    if cache_path is not None:
        save_pickle(kg, cache_path)
    return kg


def sanitize_kg(kg: KG, sanitize_cfg: SanitizeConfig) -> Tuple[KG, Dict[str, Any]]:
    sanitizer = ReviewKGPatternSanitizer(kg, sanitize_cfg)
    return sanitizer.sanitize()


def inject_fraud_into_kg(kg: KG, fraud_cfg: FraudInjectionConfig) -> Tuple[KG, Dict[str, Any]]:
    injector = ReviewKGUserFraudInjector(kg, fraud_cfg)
    stats = injector.inject()
    return kg, stats


def build_training_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        fixed_threshold=args.fixed_threshold,
        select_metric=args.select_metric,
    )


def train_on_synthetic_and_evaluate_transfer(
    synthetic_fraud_kg: KG,
    real_fraud_kg: KG,
    args,
) -> Dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    task_cfg = TaskConfig(
        target_node_type="User",
        label_property="fraud",
        positive_value=True,
    )
    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratified=True,
    )
    train_args = build_training_args(args)

    synthetic_data = build_heterodata_generic(synthetic_fraud_kg, task_cfg)
    y = synthetic_data[task_cfg.target_node_type].y
    train_mask, val_mask, test_mask = make_single_split_masks(y, split_cfg)
    assign_masks(synthetic_data, task_cfg.target_node_type, train_mask, val_mask, test_mask)
    validate_masks(synthetic_data, task_cfg.target_node_type)

    train_result = run_one_training(synthetic_data, task_cfg, train_args, device)

    real_data = build_heterodata_generic(real_fraud_kg, task_cfg)
    model = HeteroGenericSAGE(
        real_data.metadata(),
        hidden_dim=args.hidden_dim,
        out_dim=2,
        dropout=args.dropout,
    ).to(device)

    real_data = real_data.to(device)
    with torch.no_grad():
        _ = model(real_data.x_dict, real_data.edge_index_dict)
    model.load_state_dict(train_result["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(real_data.x_dict, real_data.edge_index_dict)[task_cfg.target_node_type]
        y_real = real_data[task_cfg.target_node_type].y
        pos_probs = torch.softmax(logits, dim=-1)[:, 1]
        precision, recall, f1 = _binary_scores_from_probs(pos_probs, y_real, train_result["threshold"])
        auc = _binary_auc_from_probs(pos_probs, y_real)

    transfer_metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(train_result["threshold"]),
        "num_users_evaluated": int(y_real.numel()),
        "num_positive_users": int((y_real == 1).sum().item()),
        "num_negative_users": int((y_real == 0).sum().item()),
    }

    synthetic_test_metrics = {k: float(v) for k, v in train_result["metrics"]["test"].items()}
    synthetic_train_metrics = {k: float(v) for k, v in train_result["metrics"]["train"].items()}
    synthetic_val_metrics = {k: float(v) for k, v in train_result["metrics"]["val"].items()}

    return {
        "synthetic_train_metrics": synthetic_train_metrics,
        "synthetic_val_metrics": synthetic_val_metrics,
        "synthetic_test_metrics": synthetic_test_metrics,
        "transfer_real_metrics": transfer_metrics,
        "chosen_threshold": float(train_result["threshold"]),
        "model_state_dict_path": None,
        "synthetic_split_counts": {
            "train": {
                "size": int(train_mask.sum().item()),
                "positives": int(y[train_mask].sum().item()),
                "negatives": int((y[train_mask] == 0).sum().item()),
            },
            "val": {
                "size": int(val_mask.sum().item()),
                "positives": int(y[val_mask].sum().item()),
                "negatives": int((y[val_mask] == 0).sum().item()),
            },
            "test": {
                "size": int(test_mask.sum().item()),
                "positives": int(y[test_mask].sum().item()),
                "negatives": int((y[test_mask] == 0).sum().item()),
            },
        },
    }


def run_one_size(size_name: str, category: str, args) -> Dict[str, Any]:
    out_dir = Path(args.output_dir) / size_name
    ensure_dir(out_dir)

    original_cache = out_dir / "original_clean_kg.pkl" if args.cache_pickles else None
    synthetic_cache = out_dir / "synthetic_clean_kg_optimized.pkl" if args.cache_pickles else None

    print(f"\n==================== {size_name.upper()} / {category} ====================")
    print("Building original KG...")
    original_clean = maybe_cache_build_original_kg(category, original_cache)

    print("Inferring schema...")
    schema = infer_schema_from_kg(
        original_clean,
        seed=args.seed,
        no_duplicate_triples=True,
    )
    schema_json_path = out_dir / f"{size_name}_schema.json"
    schema_txt_path = out_dir / f"{size_name}_schema.txt"
    save_json(schema, schema_json_path)
    schema_txt_path.write_text(schema_to_pseudo_text(schema), encoding="utf-8")

    print("Generating clean synthetic KG with optimized generator...")
    synthetic_clean = maybe_cache_generate_synthetic(schema_json_path, synthetic_cache)

    print("Evaluating clean synthetic fidelity...")
    fidelity_report = build_evaluation_report(
        original_clean,
        synthetic_clean,
        category=category,
        schema_path=str(schema_json_path),
    )
    save_json(fidelity_report, out_dir / "clean_fidelity_report.json")

    clean_original_summary = summarize_kg(original_clean)
    clean_synthetic_summary = summarize_kg(synthetic_clean)

    sanitize_cfg = default_sanitize_cfg(args.seed)
    fraud_cfg = default_fraud_cfg(args.seed, args.corruption_rate)

    print("Sanitizing clean synthetic KG...")
    synthetic_sanitized, synthetic_sanitize_stats = sanitize_kg(synthetic_clean, sanitize_cfg)
    print("Injecting fraud into synthetic KG in place...")
    synthetic_fraud, synthetic_fraud_stats = inject_fraud_into_kg(synthetic_sanitized, fraud_cfg)

    del synthetic_clean
    del synthetic_sanitized
    gc.collect()

    print("Injecting fraud into original KG in place...")
    real_fraud, real_fraud_stats = inject_fraud_into_kg(original_clean, fraud_cfg)
    real_sanitize_stats = {
        "status": "skipped",
        "reason": "original graph was not sanitized"
    }

    print("Training on synthetic fraud graph and evaluating transfer...")
    gnn_results = train_on_synthetic_and_evaluate_transfer(
        synthetic_fraud_kg=synthetic_fraud,
        real_fraud_kg=real_fraud,
        args=args,
    )

    del synthetic_fraud
    del real_fraud
    gc.collect()

    result = {
        "size_name": size_name,
        "category": category,
        "paths": {
            "schema_json": str(schema_json_path),
            "schema_pseudograph": str(schema_txt_path),
            "clean_fidelity_report": str(out_dir / "clean_fidelity_report.json"),
        },
        "clean_original_summary": clean_original_summary,
        "clean_synthetic_summary": clean_synthetic_summary,
        "synthetic_sanitize_stats": synthetic_sanitize_stats,
        "synthetic_fraud_injection_stats": synthetic_fraud_stats,
        "real_sanitize_stats": real_sanitize_stats,
        "real_fraud_injection_stats": real_fraud_stats,
        "fidelity_report_excerpt": {
            "global": fidelity_report["global"],
            "node_counts_by_label": fidelity_report["node_counts_by_label"],
            "edge_counts_by_type": fidelity_report["edge_counts_by_type"],
        },
        "gnn_results": gnn_results,
    }

    save_json(result, out_dir / "report_summary.json")
    return result


def build_master_summary(per_size_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def pick(res: Dict[str, Any], *keys):
        cur = res
        for k in keys:
            cur = cur[k]
        return cur

    summary = {"sizes": {}, "tables_for_report": {}}
    for res in per_size_results:
        size = res["size_name"]
        summary["sizes"][size] = res

    summary["tables_for_report"]["clean_fidelity_global"] = {
        res["size_name"]: {
            "node_count_relative_error": pick(res, "fidelity_report_excerpt", "global", "node_count_relative_error"),
            "edge_count_relative_error": pick(res, "fidelity_report_excerpt", "global", "edge_count_relative_error"),
        }
        for res in per_size_results
    }

    summary["tables_for_report"]["synthetic_model_results"] = {
        res["size_name"]: pick(res, "gnn_results", "synthetic_test_metrics")
        for res in per_size_results
    }

    summary["tables_for_report"]["transfer_results"] = {
        res["size_name"]: pick(res, "gnn_results", "transfer_real_metrics")
        for res in per_size_results
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Amazon synthetic-KG report pipeline for small, medium, and large categories."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["small", "medium", "large", "all"],
        default=["all"],
        help="Which size configurations to run.",
    )
    parser.add_argument("--output-dir", type=str, default="report_pipeline_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corruption-rate", type=float, default=0.05)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--fixed-threshold", type=float, default=0.85)
    parser.add_argument("--select-metric", choices=["f1", "auc"], default="f1")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--cache-pickles",
        action="store_true",
        help="Cache original and synthetic clean KGs as pickles for faster reruns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    size_list = ["small", "medium", "large"] if "all" in args.sizes else args.sizes
    ensure_dir(Path(args.output_dir))

    all_results = []
    for size_name in size_list:
        curr_time = int(time.time())
        category = SIZE_TO_CATEGORY[size_name]
        res = run_one_size(size_name, category, args)
        all_results.append(res)
        print(f"Time taken for {size_name}: {(int(time.time()) - curr_time) / 60:.2f} minutes")

    master_summary = build_master_summary(all_results)
    master_path = Path(args.output_dir) / "master_report_summary.json"
    save_json(master_summary, master_path)

    print("\n==================== DONE ====================")
    print(f"Saved master summary to: {master_path}")
    for res in all_results:
        size = res["size_name"]
        test_metrics = res["gnn_results"]["synthetic_test_metrics"]
        transfer_metrics = res["gnn_results"]["transfer_real_metrics"]
        print(
            f"{size}: synthetic_test_f1={test_metrics['f1']:.4f}, "
            f"synthetic_test_auc={test_metrics['auc']:.4f}, "
            f"transfer_f1={transfer_metrics['f1']:.4f}, "
            f"transfer_auc={transfer_metrics['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
