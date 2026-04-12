from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from torch_geometric.loader import NeighborLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

from utils import KG, build_amazon_reviews_kg
from src.kg_generator import generate_kg_from_schema

try:
    from src.review_fraud_injector import FraudInjectionConfig, ReviewKGUserFraudInjector
except Exception:
    FraudInjectionConfig = None
    ReviewKGUserFraudInjector = None

try:
    from src.review_pattern_sanitizer import SanitizeConfig, ReviewKGPatternSanitizer
except Exception:
    SanitizeConfig = None
    ReviewKGPatternSanitizer = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return default


def _text_len(value: Any) -> float:
    if value is None:
        return 0.0
    return float(len(str(value)))


def _standardize(x: Tensor) -> Tensor:
    if x.numel() == 0:
        return x
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-12, torch.ones_like(std), std)
    return (x - mean) / std


def _coerce_cli_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None
    try:
        if raw.strip().isdigit() or (raw.strip().startswith("-") and raw.strip()[1:].isdigit()):
            return int(raw)
        return float(raw)
    except Exception:
        return raw


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratified: bool = True


@dataclass
class TaskConfig:
    target_node_type: str = "User"
    label_property: str = "fraud"
    positive_value: Any = True


class BaseSAGE(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return self.lin(x)


class HeteroGenericSAGE(nn.Module):
    def __init__(self, metadata, hidden_dim: int = 64, out_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        base = BaseSAGE(hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
        self.model = to_hetero(base, metadata, aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        return self.model(x_dict, edge_index_dict)


def build_graph_from_args(args) -> KG:
    if args.graph_source == "original":
        kg = build_amazon_reviews_kg(args.category)
    else:
        kg = generate_kg_from_schema(args.schema)

    if args.sanitize:
        if ReviewKGPatternSanitizer is None or SanitizeConfig is None:
            raise RuntimeError("Sanitizer modules are not available in this environment.")
        sanitize_cfg = SanitizeConfig(
            seed=args.seed,
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
        kg, sanitize_stats = ReviewKGPatternSanitizer(kg, sanitize_cfg).sanitize()
        print("SANITIZE:", sanitize_stats)

    if args.inject_fraud:
        if ReviewKGUserFraudInjector is None or FraudInjectionConfig is None:
            raise RuntimeError("Fraud injector modules are not available in this environment.")
        fraud_cfg = FraudInjectionConfig(
            seed=args.seed,
            corruption_rate=args.corruption_rate,
            criminal_user_fraction=args.criminal_user_fraction,
            min_criminal_users=args.min_criminal_users,
            camouflage_rate=args.camouflage_rate,
        )
        inject_stats = ReviewKGUserFraudInjector(kg, fraud_cfg).inject()
        print("INJECT:", inject_stats)

    return kg


def _infer_prop_kind(values: Sequence[Any]) -> str:
    seen_numeric = False
    seen_bool = False
    seen_text = False
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            seen_bool = True
        elif isinstance(v, (int, float)):
            seen_numeric = True
        else:
            seen_text = True
    if seen_text:
        return "text"
    if seen_numeric:
        return "numeric"
    if seen_bool:
        return "bool"
    return "unknown"


def _build_degree_maps(
    kg: KG,
) -> Tuple[
    Dict[Tuple[str, str], Dict[int, int]],
    Dict[Tuple[str, str], Dict[int, int]],
    Dict[str, List[str]],
    Dict[str, List[str]],
]:
    out_deg: Dict[Tuple[str, str], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    in_deg: Dict[Tuple[str, str], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    rels_out: Dict[str, set] = defaultdict(set)
    rels_in: Dict[str, set] = defaultdict(set)

    for rel_type, edges in kg.edges_by_type.items():
        for e in edges:
            src_type = kg.node_label[e.src]
            dst_type = kg.node_label[e.dst]
            out_deg[(src_type, rel_type)][e.src] += 1
            in_deg[(dst_type, rel_type)][e.dst] += 1
            rels_out[src_type].add(rel_type)
            rels_in[dst_type].add(rel_type)

    rels_out_sorted = {k: sorted(v) for k, v in rels_out.items()}
    rels_in_sorted = {k: sorted(v) for k, v in rels_in.items()}
    return out_deg, in_deg, rels_out_sorted, rels_in_sorted


def _extract_node_type_features(
    kg: KG,
    node_type: str,
    node_ids: List[int],
    out_deg: Dict[Tuple[str, str], Dict[int, int]],
    in_deg: Dict[Tuple[str, str], Dict[int, int]],
    rels_out: Dict[str, List[str]],
    rels_in: Dict[str, List[str]],
    task_cfg: TaskConfig,
) -> Tuple[Tensor, List[str]]:
    raw_prop_names = sorted({
        pname
        for nid in node_ids
        for pname in kg.node_props.get(nid, {}).keys()
        if pname != "id"
        and not (
            node_type == task_cfg.target_node_type
            and pname == task_cfg.label_property
        )
    })

    prop_kinds: Dict[str, str] = {}
    for pname in raw_prop_names:
        values = [kg.node_props.get(nid, {}).get(pname) for nid in node_ids]
        prop_kinds[pname] = _infer_prop_kind(values)

    prop_names = [pname for pname in raw_prop_names if prop_kinds[pname] != "text"]

    feature_names: List[str] = []
    rows: List[List[float]] = []
    out_rel_list = rels_out.get(node_type, [])
    in_rel_list = rels_in.get(node_type, [])

    for pname in prop_names:
        kind = prop_kinds[pname]
        if kind == "numeric":
            feature_names.append(f"prop:{pname}")
        elif kind == "bool":
            feature_names.append(f"prop:{pname}")
        else:
            feature_names.append(f"prop_present:{pname}")

    feature_names.append("deg_out_total")
    feature_names.append("deg_in_total")
    for rel in out_rel_list:
        feature_names.append(f"deg_out:{rel}")
    for rel in in_rel_list:
        feature_names.append(f"deg_in:{rel}")

    for nid in node_ids:
        props = kg.node_props.get(nid, {})
        row: List[float] = []

        for pname in prop_names:
            kind = prop_kinds[pname]
            value = props.get(pname)
            if kind == "numeric":
                row.append(_safe_float(value, 0.0))
            elif kind == "bool":
                row.append(1.0 if bool(value) else 0.0)
            else:
                row.append(0.0 if value is None else 1.0)

        total_out = 0.0
        for rel in out_rel_list:
            total_out += float(out_deg[(node_type, rel)].get(nid, 0))
        total_in = 0.0
        for rel in in_rel_list:
            total_in += float(in_deg[(node_type, rel)].get(nid, 0))

        row.append(total_out)
        row.append(total_in)

        for rel in out_rel_list:
            row.append(float(out_deg[(node_type, rel)].get(nid, 0)))
        for rel in in_rel_list:
            row.append(float(in_deg[(node_type, rel)].get(nid, 0)))

        if not row:
            row = [0.0]
            feature_names[:] = ["bias"]
        rows.append(row)

    x = torch.tensor(rows, dtype=torch.float)
    return _standardize(x), feature_names


def _make_binary_labels(node_ids: List[int], kg: KG, task_cfg: TaskConfig) -> Tensor:
    y: List[int] = []
    pos_value = task_cfg.positive_value
    for gid in node_ids:
        props = kg.node_props.get(gid, {})
        value = props.get(task_cfg.label_property)
        y.append(1 if value == pos_value else 0)
    return torch.tensor(y, dtype=torch.long)


def build_heterodata_generic(kg: KG, task_cfg: TaskConfig) -> HeteroData:
    if task_cfg.target_node_type not in kg.nodes_by_label:
        raise ValueError(
            f"Target node type '{task_cfg.target_node_type}' not found. "
            f"Available types: {sorted(kg.nodes_by_label.keys())}"
        )

    data = HeteroData()

    type_to_global_ids: Dict[str, List[int]] = {
        node_type: list(ids)
        for node_type, ids in kg.nodes_by_label.items()
    }
    global_to_local: Dict[str, Dict[int, int]] = {
        node_type: {gid: i for i, gid in enumerate(ids)}
        for node_type, ids in type_to_global_ids.items()
    }

    out_deg, in_deg, rels_out, rels_in = _build_degree_maps(kg)

    feature_metadata: Dict[str, List[str]] = {}
    for node_type, ids in type_to_global_ids.items():
        x, names = _extract_node_type_features(
            kg, node_type, ids, out_deg, in_deg, rels_out, rels_in, task_cfg
        )
        data[node_type].x = x
        data[node_type].num_nodes = len(ids)
        feature_metadata[node_type] = names

        if node_type == task_cfg.target_node_type:
            data[node_type].y = _make_binary_labels(ids, kg, task_cfg)

    for rel_type, edges in kg.edges_by_type.items():
        if not edges:
            continue
        src_type = kg.node_label[edges[0].src]
        dst_type = kg.node_label[edges[0].dst]
        src_idx = [global_to_local[src_type][e.src] for e in edges]
        dst_idx = [global_to_local[dst_type][e.dst] for e in edges]
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        data[(src_type, rel_type, dst_type)].edge_index = edge_index
        data[(dst_type, f"rev_{rel_type}", src_type)].edge_index = torch.tensor([dst_idx, src_idx], dtype=torch.long)

    data.feature_metadata = feature_metadata
    return data


def _split_stats(y: Tensor, mask: Tensor) -> Dict[str, int]:
    ys = y[mask]
    pos = int((ys == 1).sum().item())
    neg = int((ys == 0).sum().item())
    return {"size": int(ys.numel()), "positives": pos, "negatives": neg}


def print_feature_summary(data: HeteroData) -> None:
    print("\n=== FEATURE SUMMARY ===")
    feature_metadata = getattr(data, "feature_metadata", {})
    for node_type in data.node_types:
        dim = int(data[node_type].x.size(-1)) if hasattr(data[node_type], "x") else 0
        names = feature_metadata.get(node_type, [])
        preview = ", ".join(names[:8])
        if len(names) > 8:
            preview += ", ..."
        print(f"{node_type}: dim={dim} features=[{preview}]")


def print_mask_summary(data: HeteroData, target_node_type: str, prefix: str = "") -> None:
    y = data[target_node_type].y.cpu()
    title = f"=== LABEL COUNTS ({prefix}) ===" if prefix else "=== LABEL COUNTS ==="
    print(f"\n{title}")
    print(f"total: positives={int((y == 1).sum())} negatives={int((y == 0).sum())}")
    for split in ("train", "val", "test"):
        stats = _split_stats(y, data[target_node_type][f"{split}_mask"].cpu())
        print(f"{split}: size={stats['size']} positives={stats['positives']} negatives={stats['negatives']}")


def _split_into_k_folds(indices: List[int], k: int) -> List[List[int]]:
    folds = [[] for _ in range(k)]
    for i, idx in enumerate(indices):
        folds[i % k].append(idx)
    return folds


def make_kfold_masks(
    y: Tensor,
    k: int,
    fold_idx: int,
    seed: int,
    val_ratio_within_train: float = 0.15,
) -> Tuple[Tensor, Tensor, Tensor]:
    rng = random.Random(seed)
    pos_idx = [i for i, v in enumerate(y.tolist()) if int(v) == 1]
    neg_idx = [i for i, v in enumerate(y.tolist()) if int(v) == 0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = _split_into_k_folds(pos_idx, k)
    neg_folds = _split_into_k_folds(neg_idx, k)

    test_idx = pos_folds[fold_idx] + neg_folds[fold_idx]

    remain_pos = [x for i, fold in enumerate(pos_folds) if i != fold_idx for x in fold]
    remain_neg = [x for i, fold in enumerate(neg_folds) if i != fold_idx for x in fold]

    rng.shuffle(remain_pos)
    rng.shuffle(remain_neg)

    n_val_pos = int(val_ratio_within_train * len(remain_pos))
    n_val_neg = int(val_ratio_within_train * len(remain_neg))

    val_idx = remain_pos[:n_val_pos] + remain_neg[:n_val_neg]
    train_idx = remain_pos[n_val_pos:] + remain_neg[n_val_neg:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    n = y.size(0)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def make_single_split_masks(y: Tensor, split_cfg: SplitConfig) -> Tuple[Tensor, Tensor, Tensor]:
    n = y.size(0)
    if split_cfg.stratified:
        rng = random.Random(split_cfg.seed)
        pos_idx = [i for i, v in enumerate(y.tolist()) if int(v) == 1]
        neg_idx = [i for i, v in enumerate(y.tolist()) if int(v) == 0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        def split_one(indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
            n_local = len(indices)
            n_train = int(split_cfg.train_ratio * n_local)
            n_val = int(split_cfg.val_ratio * n_local)
            return (
                indices[:n_train],
                indices[n_train:n_train + n_val],
                indices[n_train + n_val:],
            )

        pos_train, pos_val, pos_test = split_one(pos_idx)
        neg_train, neg_val, neg_test = split_one(neg_idx)

        train_idx = pos_train + neg_train
        val_idx = pos_val + neg_val
        test_idx = pos_test + neg_test
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
    else:
        indices = list(range(n))
        rng = random.Random(split_cfg.seed)
        rng.shuffle(indices)
        n_train = int(split_cfg.train_ratio * n)
        n_val = int(split_cfg.val_ratio * n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def assign_masks(data: HeteroData, target_node_type: str, train_mask: Tensor, val_mask: Tensor, test_mask: Tensor) -> None:
    data[target_node_type].train_mask = train_mask
    data[target_node_type].val_mask = val_mask
    data[target_node_type].test_mask = test_mask


def validate_masks(data: HeteroData, target_node_type: str) -> None:
    y = data[target_node_type].y
    total_pos = int((y == 1).sum().item())
    if total_pos == 0:
        raise RuntimeError(f"No positive labels found on target node type '{target_node_type}'.")
    for split in ("train", "val", "test"):
        stats = _split_stats(y, data[target_node_type][f"{split}_mask"])
        if stats["positives"] == 0 or stats["negatives"] == 0:
            raise RuntimeError(
                f"Split '{split}' is missing a class: {stats}. "
                f"Use stratified split or adjust k / val_ratio."
            )


def _binary_scores_from_probs(pos_probs: Tensor, y_true: Tensor, threshold: float) -> Tuple[float, float, float]:
    pred = (pos_probs >= threshold).long()
    tp = int(((pred == 1) & (y_true == 1)).sum().item())
    fp = int(((pred == 1) & (y_true == 0)).sum().item())
    fn = int(((pred == 0) & (y_true == 1)).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _binary_auc_from_probs(scores: Tensor, y_true: Tensor) -> float:
    scores = scores.detach().cpu().float()
    labels = y_true.detach().cpu().long()

    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    n = sorted_scores.numel()
    ranks = torch.empty(n, dtype=torch.float32)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  
        ranks[i:j] = avg_rank
        i = j

    sum_pos_ranks = ranks[sorted_labels == 1].sum().item()

    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _best_threshold_from_probs(pos_probs: Tensor, y_true: Tensor) -> Tuple[float, float]:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in [i / 100.0 for i in range(5, 96, 5)]:
        _, _, f1 = _binary_scores_from_probs(pos_probs, y_true, thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1

def _parse_num_neighbors(raw: Any) -> List[int]:
    if isinstance(raw, (list, tuple)):
        vals = [int(v) for v in raw]
    else:
        text = str(raw).strip()
        vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        vals = [10, 5]
    return vals


def _make_num_neighbors_dict(data: HeteroData, args) -> Dict[Any, List[int]]:
    vals = _parse_num_neighbors(getattr(args, "num_neighbors", "10,5"))
    return {edge_type: vals[:] for edge_type in data.edge_types}


def _mask_to_input_nodes(mask: Tensor) -> Tensor:
    return mask.nonzero(as_tuple=False).view(-1).cpu()


def _make_neighbor_loader(
    data: HeteroData,
    target_node_type: str,
    node_indices: Tensor,
    batch_size: int,
    num_neighbors: Dict[Any, List[int]],
    shuffle: bool,
) -> NeighborLoader:
    return NeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        input_nodes=(target_node_type, node_indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )


@torch.no_grad()
def _collect_probs_and_labels(
    model: nn.Module,
    loader: NeighborLoader,
    target_node_type: str,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    model.eval()
    probs_all: List[Tensor] = []
    labels_all: List[Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        out_dict = model(batch.x_dict, batch.edge_index_dict)
        out = out_dict[target_node_type]

        seed_count = int(batch[target_node_type].batch_size)
        probs = torch.softmax(out[:seed_count], dim=-1)[:, 1].detach().cpu()
        labels = batch[target_node_type].y[:seed_count].detach().cpu()

        probs_all.append(probs)
        labels_all.append(labels)

        del batch
        del out_dict
        del out

    if not probs_all:
        return torch.empty(0, dtype=torch.float), torch.empty(0, dtype=torch.long)

    return torch.cat(probs_all, dim=0), torch.cat(labels_all, dim=0)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    split_loaders: Dict[str, NeighborLoader],
    target_node_type: str,
    device: torch.device,
    threshold: Optional[float] = None,
    tune_on: Optional[str] = "val",
):
    split_outputs: Dict[str, Tuple[Tensor, Tensor]] = {}

    for split_name, loader in split_loaders.items():
        probs, labels = _collect_probs_and_labels(model, loader, target_node_type, device)
        split_outputs[split_name] = (probs, labels)

    chosen_threshold = 0.5
    if threshold is not None:
        chosen_threshold = threshold
    elif tune_on is not None:
        tune_probs, tune_labels = split_outputs[tune_on]
        chosen_threshold, _ = _best_threshold_from_probs(tune_probs, tune_labels)

    result = {}
    for split_name, (split_probs, labels) in split_outputs.items():
        precision, recall, f1 = _binary_scores_from_probs(split_probs, labels, chosen_threshold)
        auc = _binary_auc_from_probs(split_probs, labels)
        result[split_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

    return result, chosen_threshold


def run_one_training(
    data: HeteroData,
    task_cfg: TaskConfig,
    args,
    device: torch.device,
    fold_name: str = "",
) -> Dict[str, Any]:
    torch.set_num_threads(1)

    target_node_type = task_cfg.target_node_type
    train_batch_size = int(getattr(args, "batch_size", 512))
    eval_batch_size = int(getattr(args, "eval_batch_size", 1024))
    num_neighbors = _make_num_neighbors_dict(data, args)

    train_input_nodes = _mask_to_input_nodes(data[target_node_type].train_mask)
    val_input_nodes = _mask_to_input_nodes(data[target_node_type].val_mask)
    test_input_nodes = _mask_to_input_nodes(data[target_node_type].test_mask)

    train_loader = _make_neighbor_loader(
        data=data,
        target_node_type=target_node_type,
        node_indices=train_input_nodes,
        batch_size=train_batch_size,
        num_neighbors=num_neighbors,
        shuffle=True,
    )

    eval_loaders = {
        "train": _make_neighbor_loader(
            data=data,
            target_node_type=target_node_type,
            node_indices=train_input_nodes,
            batch_size=eval_batch_size,
            num_neighbors=num_neighbors,
            shuffle=False,
        ),
        "val": _make_neighbor_loader(
            data=data,
            target_node_type=target_node_type,
            node_indices=val_input_nodes,
            batch_size=eval_batch_size,
            num_neighbors=num_neighbors,
            shuffle=False,
        ),
        "test": _make_neighbor_loader(
            data=data,
            target_node_type=target_node_type,
            node_indices=test_input_nodes,
            batch_size=eval_batch_size,
            num_neighbors=num_neighbors,
            shuffle=False,
        ),
    }

    model = HeteroGenericSAGE(
        data.metadata(),
        hidden_dim=args.hidden_dim,
        out_dim=2,
        dropout=args.dropout,
    ).to(device)

    first_batch = next(iter(train_loader))
    first_batch = first_batch.to(device)
    with torch.no_grad():
        _ = model(first_batch.x_dict, first_batch.edge_index_dict)
    del first_batch

    y_train = data[target_node_type].y[data[target_node_type].train_mask].cpu()
    pos = int((y_train == 1).sum().item())
    neg = int((y_train == 0).sum().item())
    weight = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float, device=device) if pos > 0 else None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_select_metric = -1.0
    best_aux_metric = -1.0
    best_state = None
    best_threshold = 0.5
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            out_dict = model(batch.x_dict, batch.edge_index_dict)
            out = out_dict[target_node_type]

            seed_count = int(batch[target_node_type].batch_size)
            y = batch[target_node_type].y[:seed_count]

            loss = criterion(out[:seed_count], y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * seed_count
            total_examples += seed_count

            del batch
            del out_dict
            del out
            del y
            del loss

        epoch_loss = total_loss / max(total_examples, 1)

        if args.fixed_threshold >= 0.0:
            metrics, chosen_thr = evaluate(
                model=model,
                split_loaders=eval_loaders,
                target_node_type=target_node_type,
                device=device,
                threshold=args.fixed_threshold,
                tune_on=None,
            )
        else:
            metrics, chosen_thr = evaluate(
                model=model,
                split_loaders=eval_loaders,
                target_node_type=target_node_type,
                device=device,
                threshold=None,
                tune_on="val",
            )

        prefix = f"[{fold_name}] " if fold_name else ""
        print(
            f"{prefix}epoch={epoch:03d} "
            f"loss={epoch_loss:.4f} "
            f"val_auc={metrics['val']['auc']:.4f} "
            f"val_f1={metrics['val']['f1']:.4f} "
            f"thr={chosen_thr:.2f} "
            f"test_f1={metrics['test']['f1']:.4f}"
        )

        current_val_f1 = metrics["val"]["f1"]
        current_val_auc = metrics["val"]["auc"]

        if args.select_metric == "f1":
            select_value = current_val_f1
            aux_value = -1.0 if math.isnan(current_val_auc) else current_val_auc
        else:
            select_value = -1.0 if math.isnan(current_val_auc) else current_val_auc
            aux_value = current_val_f1

        improved = False
        if select_value > best_select_metric + 1e-12:
            improved = True
        elif abs(select_value - best_select_metric) <= 1e-12 and aux_value > best_aux_metric + 1e-12:
            improved = True

        if improved:
            best_select_metric = select_value
            best_aux_metric = aux_value
            best_threshold = chosen_thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        final_state = best_state
    else:
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    use_threshold = args.fixed_threshold if args.fixed_threshold >= 0.0 else best_threshold
    final_metrics, _ = evaluate(
        model=model,
        split_loaders=eval_loaders,
        target_node_type=target_node_type,
        device=device,
        threshold=use_threshold,
        tune_on=None,
    )

    return {
        "state_dict": final_state,
        "threshold": use_threshold,
        "metrics": final_metrics,
    }


def train_single_split(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    task_cfg = TaskConfig(
        target_node_type=args.target_node_type,
        label_property=args.label_property,
        positive_value=_coerce_cli_value(args.positive_value),
    )
    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratified=(not args.no_stratified_split),
    )

    kg = build_graph_from_args(args)
    data = build_heterodata_generic(kg, task_cfg)
    print_feature_summary(data)

    y = data[task_cfg.target_node_type].y
    train_mask, val_mask, test_mask = make_single_split_masks(y, split_cfg)
    assign_masks(data, task_cfg.target_node_type, train_mask, val_mask, test_mask)
    print_mask_summary(data, task_cfg.target_node_type)
    validate_masks(data, task_cfg.target_node_type)

    result = run_one_training(data, task_cfg, args, device)

    print("\n=== FINAL ===")
    print(
        f"target_node_type={task_cfg.target_node_type} "
        f"label_property={task_cfg.label_property} "
        f"positive_value={task_cfg.positive_value!r}"
    )
    print(f"threshold={result['threshold']:.2f}")
    for split in ("train", "val", "test"):
        m = result["metrics"][split]
        print(
            f"{split}: "
            f"precision={m['precision']:.4f} "
            f"recall={m['recall']:.4f} "
            f"f1={m['f1']:.4f} "
            f"auc={m['auc']:.4f}"
        )

    if args.save_model:
        torch.save({
            "state_dict": result["state_dict"],
            "threshold": result["threshold"],
            "target_node_type": task_cfg.target_node_type,
            "label_property": task_cfg.label_property,
            "positive_value": task_cfg.positive_value,
        }, args.save_model)
        print(f"Saved model to: {args.save_model}")


def train_kfold(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if args.k_folds < 2:
        raise ValueError("--k-folds must be at least 2 when using --use-kfold")

    task_cfg = TaskConfig(
        target_node_type=args.target_node_type,
        label_property=args.label_property,
        positive_value=_coerce_cli_value(args.positive_value),
    )

    kg = build_graph_from_args(args)
    base_data = build_heterodata_generic(kg, task_cfg)
    print_feature_summary(base_data)

    y = base_data[task_cfg.target_node_type].y
    total_pos = int((y == 1).sum().item())
    if total_pos == 0:
        raise RuntimeError(
            f"No positive labels found on target node type '{task_cfg.target_node_type}' "
            f"using label property '{task_cfg.label_property}' == {task_cfg.positive_value!r}."
        )

    fold_test_metrics: List[Dict[str, float]] = []
    fold_thresholds: List[float] = []

    for fold_idx in range(args.k_folds):
        train_mask, val_mask, test_mask = make_kfold_masks(
            y=y.cpu(),
            k=args.k_folds,
            fold_idx=fold_idx,
            seed=args.seed,
            val_ratio_within_train=args.kfold_val_ratio,
        )

        data = base_data.clone()
        assign_masks(data, task_cfg.target_node_type, train_mask, val_mask, test_mask)
        print_mask_summary(data, task_cfg.target_node_type, prefix=f"fold {fold_idx + 1}/{args.k_folds}")
        validate_masks(data, task_cfg.target_node_type)

        result = run_one_training(
            data=data,
            task_cfg=task_cfg,
            args=args,
            device=device,
            fold_name=f"fold {fold_idx + 1}/{args.k_folds}",
        )

        test_metrics = result["metrics"]["test"]
        fold_test_metrics.append(test_metrics)
        fold_thresholds.append(result["threshold"])

        print(f"\n=== FOLD {fold_idx + 1}/{args.k_folds} FINAL ===")
        print(f"threshold={result['threshold']:.2f}")
        print(
            f"test: precision={test_metrics['precision']:.4f} "
            f"recall={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1']:.4f} "
            f"auc={test_metrics['auc']:.4f}"
        )

    print("\n=== K-FOLD TEST SUMMARY ===")
    print(
        f"target_node_type={task_cfg.target_node_type} "
        f"label_property={task_cfg.label_property} "
        f"positive_value={task_cfg.positive_value!r}"
    )
    thr_mean, thr_std = _mean_std(fold_thresholds)
    print(f"threshold: mean={thr_mean:.4f} std={thr_std:.4f}")

    for metric_name in ("precision", "recall", "f1", "auc"):
        vals = [m[metric_name] for m in fold_test_metrics]
        mean, std = _mean_std(vals)
        print(f"{metric_name}: mean={mean:.4f} std={std:.4f}")

    if args.save_model:
        print("Note: --save-model is ignored in k-fold mode because there are multiple trained fold models.")


def train(args) -> None:
    if args.use_kfold:
        train_kfold(args)
    else:
        train_single_split(args)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a generic hetero GNN on a KG with automatic graph discovery.")
    parser.add_argument("--graph-source", choices=["original", "synthetic"], default="synthetic")
    parser.add_argument("--category", default="raw_review_Digital_Music")
    parser.add_argument("--schema", default="report_pipeline_outputs/small/small_schema.json")

    parser.add_argument("--target-node-type", default="User")
    parser.add_argument("--label-property", default="fraud")
    parser.add_argument("--positive-value", default="true")

    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--inject-fraud", action="store_true")
    parser.add_argument("--corruption-rate", type=float, default=0.05)
    parser.add_argument("--criminal-user-fraction", type=float, default=0.01)
    parser.add_argument("--min-criminal-users", type=int, default=200)
    parser.add_argument("--camouflage-rate", type=float, default=0.30)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    parser.add_argument("--use-kfold", action="store_true")
    parser.add_argument("--k-folds", type=int, default=10)
    parser.add_argument("--kfold-val-ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-stratified-split", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-model", default="")
    parser.add_argument("--select-metric", choices=["f1", "auc"], default="f1")
    parser.add_argument("--fixed-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
