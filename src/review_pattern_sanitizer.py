
import math
import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class SanitizeConfig:
    seed: int = 42

    min_reviews_repeated_star: int = 6
    dominant_star_ratio_threshold: float = 0.85

    min_reviews_deviation: int = 4
    avg_abs_deviation_threshold: float = 1.25

    min_reviews_group_concentration: int = 6
    max_group_concentration_threshold: float = 0.80

    min_same_product_block_size: int = 3
    same_product_block_share_threshold: float = 0.60

    enable_overlap_groups: bool = False

    min_common_products_overlap: int = 3
    min_jaccard_overlap: float = 0.60
    min_overlap_component_size: int = 3

    overlap_max_product_users: int = 100

    min_user_score_to_remove: int = 2

    max_removed_user_fraction: float = 0.05


class ReviewKGPatternSanitizer:
    def __init__(self, kg, cfg: SanitizeConfig):
        self.kg = kg
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.num_users = len(self.kg.nodes_by_label.get("User", []))

        self.review_to_user: Dict[int, int] = {}
        self.review_to_product: Dict[int, int] = {}
        self.review_rating: Dict[int, float] = {}

        self.user_review_count: Counter = Counter()
        self.user_products: Dict[int, Set[int]] = defaultdict(set)

        self.product_review_count: Counter = Counter()
        self.product_users: Dict[int, Set[int]] = defaultdict(set)

        self.product_to_group: Dict[int, int] = {}

        self.user_score: Counter = Counter()
        self.user_reasons: Dict[int, List[str]] = defaultdict(list)

        self.user_star_counts: Dict[int, Counter] = defaultdict(Counter)
        self.user_group_counts: Dict[int, Counter] = defaultdict(Counter)
        self.user_dev_sum: Dict[int, float] = defaultdict(float)
        self.user_dev_count: Dict[int, int] = defaultdict(int)
        self.product_mean_rating: Dict[int, float] = {}
        self.product_extreme_star_users: Dict[int, Dict[float, List[int]]] = defaultdict(lambda: defaultdict(list))

        self._build_indices()

    def sanitize(self):
        self._flag_repeated_star_users()
        self._flag_deviation_users()
        self._flag_group_concentrated_users()
        self._flag_same_product_blocks()
        if self.cfg.enable_overlap_groups:
            self._flag_overlap_groups()

        users_to_remove = {
            uid for uid, score in self.user_score.items()
            if score >= self.cfg.min_user_score_to_remove
        }

        max_remove = int(math.floor(self.cfg.max_removed_user_fraction * max(1, self.num_users)))
        max_remove = max(max_remove, 1)

        if len(users_to_remove) > max_remove:
            ranked = sorted(
                users_to_remove,
                key=lambda u: (self.user_score[u], len(self.user_products.get(u, ()))),
                reverse=True,
            )
            users_to_remove = set(ranked[:max_remove])

        new_kg, rebuild_stats = rebuild_kg_excluding_users_and_their_reviews_fast(
            self.kg,
            users_to_remove,
            self.review_to_user,
        )

        stats = {
            "users_flagged_before_cap": len([
                u for u, s in self.user_score.items()
                if s >= self.cfg.min_user_score_to_remove
            ]),
            "users_removed": rebuild_stats["users_removed"],
            "reviews_removed": rebuild_stats["reviews_removed"],
            "max_removed_user_fraction": self.cfg.max_removed_user_fraction,
            "overlap_groups_enabled": bool(self.cfg.enable_overlap_groups),
            "top_removed_users": self._top_removed_users(users_to_remove, top_k=20),
        }

        return new_kg, stats

    def _build_indices(self) -> None:
        wrote_edges = self.kg.edges_by_type.get("WROTE", [])
        about_edges = self.kg.edges_by_type.get("ABOUT", [])
        belongs_edges = self.kg.edges_by_type.get("BELONGS_TO", [])
        node_props = self.kg.node_props

        for e in belongs_edges:
            self.product_to_group[e.src] = e.dst

        for e in wrote_edges:
            self.review_to_user[e.dst] = e.src
            self.user_review_count[e.src] += 1

        for e in about_edges:
            self.review_to_product[e.src] = e.dst

        product_rating_sum: Dict[int, float] = defaultdict(float)
        product_rating_count: Dict[int, int] = defaultdict(int)
        enable_overlap = bool(self.cfg.enable_overlap_groups)

        for rid, uid in self.review_to_user.items():
            pid = self.review_to_product.get(rid)
            if pid is None:
                continue

            self.user_products[uid].add(pid)
            self.product_review_count[pid] += 1
            if enable_overlap:
                self.product_users[pid].add(uid)

            rating = node_props.get(rid, {}).get("rating")
            if rating is None:
                continue

            try:
                r = float(rating)
            except Exception:
                continue

            self.review_rating[rid] = r
            self.user_star_counts[uid][r] += 1
            product_rating_sum[pid] += r
            product_rating_count[pid] += 1

            if r in (1.0, 2.0, 4.0, 5.0):
                self.product_extreme_star_users[pid][r].append(uid)

            gid = self.product_to_group.get(pid)
            if gid is not None:
                self.user_group_counts[uid][gid] += 1

        self.product_mean_rating = {
            pid: (product_rating_sum[pid] / product_rating_count[pid])
            for pid in product_rating_count
            if product_rating_count[pid] > 0
        }

        for rid, uid in self.review_to_user.items():
            pid = self.review_to_product.get(rid)
            rating = self.review_rating.get(rid)
            mean_rating = self.product_mean_rating.get(pid)
            if pid is None or rating is None or mean_rating is None:
                continue

            self.user_dev_sum[uid] += abs(rating - mean_rating)
            self.user_dev_count[uid] += 1

    def _bump(self, uid: int, amount: int, reason: str) -> None:
        self.user_score[uid] += amount
        if reason not in self.user_reasons[uid]:
            self.user_reasons[uid].append(reason)

    def _flag_repeated_star_users(self) -> None:
        min_reviews = self.cfg.min_reviews_repeated_star
        threshold = self.cfg.dominant_star_ratio_threshold

        for uid, counts in self.user_star_counts.items():
            n = self.user_review_count.get(uid, 0)
            if n < min_reviews or not counts:
                continue

            dominant_ratio = max(counts.values()) / float(n)
            if dominant_ratio >= threshold:
                self._bump(uid, 1, "repeated_star")

    def _flag_deviation_users(self) -> None:
        min_reviews = self.cfg.min_reviews_deviation
        threshold = self.cfg.avg_abs_deviation_threshold

        for uid, cnt in self.user_dev_count.items():
            if cnt < min_reviews:
                continue
            avg_dev = self.user_dev_sum[uid] / float(cnt)
            if avg_dev >= threshold:
                self._bump(uid, 1, "high_deviation")

    def _flag_group_concentrated_users(self) -> None:
        min_reviews = self.cfg.min_reviews_group_concentration
        threshold = self.cfg.max_group_concentration_threshold

        for uid, group_counts in self.user_group_counts.items():
            n = len(self.user_products.get(uid, ()))
            if n < min_reviews or not group_counts:
                continue

            max_conc = max(group_counts.values()) / float(n)
            if max_conc >= threshold:
                self._bump(uid, 1, "group_concentration")

    def _flag_same_product_blocks(self) -> None:
        min_block = self.cfg.min_same_product_block_size
        share_threshold = self.cfg.same_product_block_share_threshold

        for pid, star_to_users in self.product_extreme_star_users.items():
            total = self.product_review_count.get(pid, 0)
            if total < min_block:
                continue

            for users in star_to_users.values():
                block_size = len(users)
                if block_size < min_block:
                    continue

                share = block_size / float(total)
                if share >= share_threshold:
                    for uid in users:
                        self._bump(uid, 1, "same_product_block")

    def _flag_overlap_groups(self) -> None:
        min_common = self.cfg.min_common_products_overlap
        min_jaccard = self.cfg.min_jaccard_overlap
        min_component = self.cfg.min_overlap_component_size
        max_product_users = max(0, int(self.cfg.overlap_max_product_users))

        adjacency: Dict[int, Set[int]] = defaultdict(set)

        eligible_users = [
            uid for uid, prods in self.user_products.items()
            if len(prods) >= min_common
        ]
        eligible_users.sort()

        for u in eligible_users:
            up = self.user_products.get(u)
            if not up:
                continue

            candidate_counts: Counter = Counter()
            for pid in up:
                users_on_product = self.product_users.get(pid)
                if not users_on_product:
                    continue
                if max_product_users and len(users_on_product) > max_product_users:
                    continue

                for v in users_on_product:
                    if v <= u:
                        continue
                    candidate_counts[v] += 1

            if not candidate_counts:
                continue

            len_up = len(up)
            for v, common in candidate_counts.items():
                if common < min_common:
                    continue

                vp = self.user_products.get(v)
                if not vp:
                    continue

                union = len_up + len(vp) - common
                if union <= 0:
                    continue

                jacc = common / float(union)
                if jacc >= min_jaccard:
                    adjacency[u].add(v)
                    adjacency[v].add(u)

        visited: Set[int] = set()
        for uid in list(adjacency.keys()):
            if uid in visited:
                continue

            stack = [uid]
            comp = []
            while stack:
                x = stack.pop()
                if x in visited:
                    continue
                visited.add(x)
                comp.append(x)
                for nbr in adjacency.get(x, ()):
                    if nbr not in visited:
                        stack.append(nbr)

            if len(comp) >= min_component:
                for u in comp:
                    self._bump(u, 2, "overlap_group")

    def _top_removed_users(self, users_to_remove: Set[int], top_k: int = 20) -> List[Dict[str, Any]]:
        ranked = sorted(
            users_to_remove,
            key=lambda u: (self.user_score[u], len(self.user_products.get(u, ()))),
            reverse=True,
        )

        out: List[Dict[str, Any]] = []
        for uid in ranked[:top_k]:
            out.append({
                "user_node_id": uid,
                "score": int(self.user_score[uid]),
                "review_count": int(self.user_review_count.get(uid, 0)),
                "product_count": len(self.user_products.get(uid, ())),
                "reasons": list(self.user_reasons.get(uid, [])),
            })
        return out


def rebuild_kg_excluding_users_and_their_reviews_fast(kg, users_to_remove: Set[int], review_to_user: Dict[int, int]):
    """
    Build a fresh KG excluding:
    - all selected users
    - all reviews written by those users

    Faster than the earlier version because it:
    - reuses review_to_user instead of rescanning WROTE
    - skips duplicate-check logic while rebuilding edges
    """

    reviews_to_remove: Set[int] = set()
    for rid, uid in review_to_user.items():
        if uid in users_to_remove:
            reviews_to_remove.add(rid)

    remove_ids = set(users_to_remove)
    remove_ids.update(reviews_to_remove)

    new_kg = kg.__class__()
    old_to_new: Dict[int, int] = {}
    node_props = kg.node_props

    add_node = new_kg.add_node
    add_edge = new_kg.add_edge

    for node in kg.nodes:
        nid = node.id
        if nid in remove_ids:
            continue

        props = dict(node_props.get(nid, {}))
        props.pop("id", None)
        old_to_new[nid] = add_node(node.label, props)

    for rel_type, edges in kg.edges_by_type.items():
        for e in edges:
            src = old_to_new.get(e.src)
            if src is None:
                continue
            dst = old_to_new.get(e.dst)
            if dst is None:
                continue

            add_edge(
                rel_type=rel_type,
                src=src,
                dst=dst,
                props=dict(e.props) if e.props else {},
                no_duplicates=False,
            )

    return new_kg, {
        "users_removed": len(users_to_remove),
        "reviews_removed": len(reviews_to_remove),
    }
