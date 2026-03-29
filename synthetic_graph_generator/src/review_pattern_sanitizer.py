import math
import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Optional


@dataclass
class SanitizeConfig:
    seed: int = 42

    # repeated star behavior
    min_reviews_repeated_star: int = 6
    dominant_star_ratio_threshold: float = 0.85

    # strong deviation from product consensus
    min_reviews_deviation: int = 4
    avg_abs_deviation_threshold: float = 1.25

    # suspicious concentration inside one product group
    min_reviews_group_concentration: int = 6
    max_group_concentration_threshold: float = 0.80

    # suspicious same-product block
    min_same_product_block_size: int = 3
    same_product_block_share_threshold: float = 0.60

    # near-duplicate neighborhoods / dense shared targeting
    min_common_products_overlap: int = 3
    min_jaccard_overlap: float = 0.60
    min_overlap_component_size: int = 3

    # final purge rule
    min_user_score_to_remove: int = 2

    # safety cap so sanitize step does not wipe too much by accident
    max_removed_user_fraction: float = 0.05


class ReviewKGPatternSanitizer:
    def __init__(self, kg, cfg: SanitizeConfig):
        self.kg = kg
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.users: List[int] = list(self.kg.nodes_by_label.get("User", []))
        self.products: List[int] = list(self.kg.nodes_by_label.get("Product", []))
        self.reviews: List[int] = list(self.kg.nodes_by_label.get("Review", []))

        self.review_to_user: Dict[int, int] = {}
        self.review_to_product: Dict[int, int] = {}

        self.user_reviews: Dict[int, Set[int]] = defaultdict(set)
        self.user_products: Dict[int, Set[int]] = defaultdict(set)

        self.product_reviews: Dict[int, Set[int]] = defaultdict(set)
        self.product_users: Dict[int, Set[int]] = defaultdict(set)

        self.product_ratings: Dict[int, List[float]] = defaultdict(list)
        self.user_ratings: Dict[int, List[float]] = defaultdict(list)

        self.product_to_group: Dict[int, int] = {}
        self.group_to_products: Dict[int, List[int]] = defaultdict(list)

        self.user_score: Counter = Counter()
        self.user_reasons: Dict[int, List[str]] = defaultdict(list)

        self._build_indices()

    def sanitize(self):
        self._flag_repeated_star_users()
        self._flag_deviation_users()
        self._flag_group_concentrated_users()
        self._flag_same_product_blocks()
        self._flag_overlap_groups()

        users_to_remove = {
            uid for uid, score in self.user_score.items()
            if score >= self.cfg.min_user_score_to_remove
        }

        # safety cap
        max_remove = int(math.floor(self.cfg.max_removed_user_fraction * max(1, len(self.users))))
        max_remove = max(max_remove, 1)

        if len(users_to_remove) > max_remove:
            ranked = sorted(
                users_to_remove,
                key=lambda u: (self.user_score[u], len(self.user_products.get(u, set()))),
                reverse=True
            )
            users_to_remove = set(ranked[:max_remove])

        new_kg, rebuild_stats = rebuild_kg_excluding_users_and_their_reviews(self.kg, users_to_remove)

        stats = {
            "users_flagged_before_cap": len([u for u, s in self.user_score.items() if s >= self.cfg.min_user_score_to_remove]),
            "users_removed": rebuild_stats["users_removed"],
            "reviews_removed": rebuild_stats["reviews_removed"],
            "max_removed_user_fraction": self.cfg.max_removed_user_fraction,
            "top_removed_users": self._top_removed_users(users_to_remove, top_k=20),
        }

        return new_kg, stats

    def _build_indices(self) -> None:
        wrote_edges = self.kg.edges_by_type.get("WROTE", [])
        about_edges = self.kg.edges_by_type.get("ABOUT", [])
        belongs_edges = self.kg.edges_by_type.get("BELONGS_TO", [])

        for e in wrote_edges:
            self.review_to_user[e.dst] = e.src
            self.user_reviews[e.src].add(e.dst)

        for e in about_edges:
            self.review_to_product[e.src] = e.dst

        for rid, uid in self.review_to_user.items():
            pid = self.review_to_product.get(rid)
            if pid is None:
                continue

            self.user_products[uid].add(pid)
            self.product_reviews[pid].add(rid)
            self.product_users[pid].add(uid)

            rating = self.kg.node_props.get(rid, {}).get("rating")
            if rating is not None:
                try:
                    r = float(rating)
                    self.product_ratings[pid].append(r)
                    self.user_ratings[uid].append(r)
                except Exception:
                    pass

        for e in belongs_edges:
            self.product_to_group[e.src] = e.dst
            self.group_to_products[e.dst].append(e.src)

    def _bump(self, uid: int, amount: int, reason: str) -> None:
        self.user_score[uid] += amount
        if reason not in self.user_reasons[uid]:
            self.user_reasons[uid].append(reason)

    def _flag_repeated_star_users(self) -> None:
        for uid, ratings in self.user_ratings.items():
            n = len(ratings)
            if n < self.cfg.min_reviews_repeated_star:
                continue

            counts = Counter(float(r) for r in ratings)
            dominant_ratio = max(counts.values()) / float(n)

            if dominant_ratio >= self.cfg.dominant_star_ratio_threshold:
                self._bump(uid, 1, "repeated_star")

    def _flag_deviation_users(self) -> None:
        for uid, review_ids in self.user_reviews.items():
            if len(review_ids) < self.cfg.min_reviews_deviation:
                continue

            devs = []
            for rid in review_ids:
                pid = self.review_to_product.get(rid)
                if pid is None:
                    continue

                rating = self.kg.node_props.get(rid, {}).get("rating")
                if rating is None:
                    continue

                product_vals = self.product_ratings.get(pid, [])
                if not product_vals:
                    continue

                try:
                    r = float(rating)
                except Exception:
                    continue

                mean_rating = sum(product_vals) / float(len(product_vals))
                devs.append(abs(r - mean_rating))

            if len(devs) < self.cfg.min_reviews_deviation:
                continue

            avg_dev = sum(devs) / float(len(devs))
            if avg_dev >= self.cfg.avg_abs_deviation_threshold:
                self._bump(uid, 1, "high_deviation")

    def _flag_group_concentrated_users(self) -> None:
        for uid, products in self.user_products.items():
            n = len(products)
            if n < self.cfg.min_reviews_group_concentration:
                continue

            group_counts = Counter()
            for pid in products:
                gid = self.product_to_group.get(pid)
                if gid is not None:
                    group_counts[gid] += 1

            if not group_counts:
                continue

            max_conc = max(group_counts.values()) / float(n)
            if max_conc >= self.cfg.max_group_concentration_threshold:
                self._bump(uid, 1, "group_concentration")

    def _flag_same_product_blocks(self) -> None:
        for pid, review_ids in self.product_reviews.items():
            total = len(review_ids)
            if total < self.cfg.min_same_product_block_size:
                continue

            star_to_users: Dict[float, List[int]] = defaultdict(list)

            for rid in review_ids:
                uid = self.review_to_user.get(rid)
                rating = self.kg.node_props.get(rid, {}).get("rating")
                if uid is None or rating is None:
                    continue
                try:
                    star = float(rating)
                except Exception:
                    continue

                if star in (1.0, 2.0, 4.0, 5.0):
                    star_to_users[star].append(uid)

            for star, users in star_to_users.items():
                block_size = len(users)
                if block_size < self.cfg.min_same_product_block_size:
                    continue

                share = block_size / float(total)
                if share >= self.cfg.same_product_block_share_threshold:
                    for uid in users:
                        self._bump(uid, 1, "same_product_block")

    def _flag_overlap_groups(self) -> None:
        pair_overlap: Counter = Counter()

        # Count common products between reviewer pairs
        for pid, users in self.product_users.items():
            user_list = sorted(users)
            m = len(user_list)
            for i in range(m):
                u = user_list[i]
                for j in range(i + 1, m):
                    v = user_list[j]
                    pair_overlap[(u, v)] += 1

        adjacency: Dict[int, Set[int]] = defaultdict(set)

        for (u, v), common in pair_overlap.items():
            if common < self.cfg.min_common_products_overlap:
                continue

            up = self.user_products.get(u, set())
            vp = self.user_products.get(v, set())
            union = len(up | vp)
            if union == 0:
                continue

            jacc = common / float(union)
            if jacc >= self.cfg.min_jaccard_overlap:
                adjacency[u].add(v)
                adjacency[v].add(u)

        visited: Set[int] = set()

        for uid in adjacency:
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
                for nbr in adjacency[x]:
                    if nbr not in visited:
                        stack.append(nbr)

            if len(comp) >= self.cfg.min_overlap_component_size:
                for u in comp:
                    self._bump(u, 2, "overlap_group")

    def _top_removed_users(self, users_to_remove: Set[int], top_k: int = 20) -> List[Dict[str, Any]]:
        ranked = sorted(
            users_to_remove,
            key=lambda u: (self.user_score[u], len(self.user_products.get(u, set()))),
            reverse=True
        )

        out = []
        for uid in ranked[:top_k]:
            out.append({
                "user_node_id": uid,
                "score": int(self.user_score[uid]),
                "review_count": len(self.user_reviews.get(uid, set())),
                "product_count": len(self.user_products.get(uid, set())),
                "reasons": list(self.user_reasons.get(uid, [])),
            })
        return out


def rebuild_kg_excluding_users_and_their_reviews(kg, users_to_remove: Set[int]):
    """
    Build a fresh KG excluding:
    - all selected users
    - all reviews written by those users

    Important:
    - internal integer node ids are rebuilt contiguously
    - external properties like user_id, asin, review_id remain unchanged
    """

    wrote_edges = kg.edges_by_type.get("WROTE", [])
    reviews_to_remove: Set[int] = set()

    for e in wrote_edges:
        if e.src in users_to_remove:
            reviews_to_remove.add(e.dst)

    remove_ids = set(users_to_remove) | reviews_to_remove

    new_kg = kg.__class__()   # keeps same KG class
    old_to_new: Dict[int, int] = {}

    # rebuild surviving nodes
    for node in kg.nodes:
        if node.id in remove_ids:
            continue

        props = dict(kg.node_props.get(node.id, {}))
        props.pop("id", None)

        new_id = new_kg.add_node(node.label, props)
        old_to_new[node.id] = new_id

    # rebuild surviving edges
    for rel_type, edges in kg.edges_by_type.items():
        for e in edges:
            if e.src in remove_ids or e.dst in remove_ids:
                continue
            if e.src not in old_to_new or e.dst not in old_to_new:
                continue

            new_kg.add_edge(
                rel_type=rel_type,
                src=old_to_new[e.src],
                dst=old_to_new[e.dst],
                props=dict(e.props),
                no_duplicates=True
            )

    return new_kg, {
        "users_removed": len(users_to_remove),
        "reviews_removed": len(reviews_to_remove),
    }