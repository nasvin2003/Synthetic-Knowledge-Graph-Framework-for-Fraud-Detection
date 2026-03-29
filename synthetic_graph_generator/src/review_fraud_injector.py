import math
import random
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class FraudInjectionConfig:
    seed: int = 42

    # malicious review budget = corruption_rate * current number of reviews
    corruption_rate: float = 0.05

    # criminal pool size
    criminal_user_fraction: float = 0.01
    min_criminal_users: int = 100

    # extra benign-looking reviews written by fraud users,
    # but not marked in any way on the review nodes
    camouflage_rate: float = 0.30

    # stronger reuse means some fraud users participate in multiple campaigns
    reuse_bias: float = 2.0

    pattern_weights: Dict[str, float] = field(default_factory=lambda: {
        "group_same_product": 1.2,
        "single_user_many_products": 1.0,
        "dense_bipartite": 1.2,
        "near_duplicate_neighborhood": 1.0,
        "repeated_behavior": 0.9,
    })

    min_group_users: int = 3
    max_group_users: int = 8

    min_dense_users: int = 3
    max_dense_users: int = 6
    min_dense_products: int = 2
    max_dense_products: int = 5
    dense_edge_prob_min: float = 0.75
    dense_edge_prob_max: float = 0.95

    min_burst_products: int = 4
    max_burst_products: int = 10

    min_dup_users: int = 2
    max_dup_users: int = 5
    min_dup_products: int = 4
    max_dup_products: int = 8

    min_repeat_users: int = 1
    max_repeat_users: int = 3
    min_repeat_products: int = 4
    max_repeat_products: int = 8

    promote_prob: float = 0.7


class ReviewKGUserFraudInjector:
    def __init__(self, kg, cfg: FraudInjectionConfig):
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

        self.criminal_pool: Set[int] = set()
        self.user_fraud_counts: Counter = Counter()
        self.user_signature_star: Dict[int, float] = {}

        self.injected_reviews = 0
        self.injected_camouflage_reviews = 0
        self.pattern_stats: Counter = Counter()

        self._build_indices()

    def inject(self) -> Dict[str, Any]:
        if not self.users or not self.products:
            return {
                "status": "no-op",
                "reason": "graph has no users or products"
            }

        self._select_criminal_pool()

        malicious_budget = int(round(self.cfg.corruption_rate * max(1, len(self.reviews))))
        remaining = malicious_budget

        safety = 0
        while remaining > 0 and safety < max(500, malicious_budget * 5):
            safety += 1
            pattern = self._choose_pattern()

            if pattern == "group_same_product":
                added = self._inject_group_same_product(remaining)
            elif pattern == "single_user_many_products":
                added = self._inject_single_user_many_products(remaining)
            elif pattern == "dense_bipartite":
                added = self._inject_dense_bipartite(remaining)
            elif pattern == "near_duplicate_neighborhood":
                added = self._inject_near_duplicate_neighborhood(remaining)
            elif pattern == "repeated_behavior":
                added = self._inject_repeated_behavior(remaining)
            else:
                added = 0

            if added > 0:
                remaining -= added
                self.pattern_stats[pattern] += added

        camouflage_budget = int(round(self.cfg.camouflage_rate * self.injected_reviews))
        self._inject_camouflage(camouflage_budget)

        return {
            "status": "ok",
            "criminal_pool_size": len(self.criminal_pool),
            "malicious_reviews_added": self.injected_reviews,
            "extra_normal_reviews_added": self.injected_camouflage_reviews,
            "final_review_count": len(self.kg.nodes_by_label.get("Review", [])),
            "pattern_review_counts": dict(self.pattern_stats),
        }

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

    def _select_criminal_pool(self) -> None:
        budget = int(round(self.cfg.corruption_rate * max(1, len(self.reviews))))

        pool_size = max(
            self.cfg.min_criminal_users,
            int(math.ceil(self.cfg.criminal_user_fraction * len(self.users))),
            int(math.ceil(budget / 6.0)),
        )
        pool_size = min(pool_size, len(self.users))

        self.criminal_pool = set(self.rng.sample(self.users, pool_size))

        for uid in self.criminal_pool:
            self.kg.node_props[uid]["fraud"] = True

            if self.rng.random() < self.cfg.promote_prob:
                self.user_signature_star[uid] = 5.0 if self.rng.random() < 0.8 else 4.0
            else:
                self.user_signature_star[uid] = 1.0 if self.rng.random() < 0.8 else 2.0

    def _choose_pattern(self) -> str:
        keys = list(self.cfg.pattern_weights.keys())
        weights = [self.cfg.pattern_weights[k] for k in keys]
        return self.rng.choices(keys, weights=weights, k=1)[0]

    def _choose_mode(self) -> str:
        return "promote" if self.rng.random() < self.cfg.promote_prob else "demote"

    def _pick_extreme_rating(self, mode: str) -> float:
        if mode == "promote":
            return 5.0 if self.rng.random() < 0.8 else 4.0
        return 1.0 if self.rng.random() < 0.8 else 2.0

    def _clip_star(self, x: float) -> float:
        x = max(1.0, min(5.0, x))
        return float(int(round(x)))

    def _product_mean_rating(self, pid: int) -> float:
        vals = self.product_ratings.get(pid, [])
        if not vals:
            return 4.0
        return sum(vals) / len(vals)

    def _weighted_pick_from_pool(self, k: int) -> List[int]:
        available = list(self.criminal_pool)
        chosen: List[int] = []

        while available and len(chosen) < k:
            weights = []
            for uid in available:
                w = 1.0 + self.cfg.reuse_bias * self.user_fraud_counts[uid]
                weights.append(w)

            uid = self.rng.choices(available, weights=weights, k=1)[0]
            chosen.append(uid)
            available.remove(uid)

        return chosen

    def _candidate_products(
        self,
        seed_pid: Optional[int] = None,
        prefer_same_group: bool = True
    ) -> List[int]:
        if seed_pid is not None and prefer_same_group:
            gid = self.product_to_group.get(seed_pid)
            if gid is not None:
                cands = list(self.group_to_products.get(gid, []))
                if cands:
                    return cands
        return list(self.products)

    def _pick_products(
        self,
        k: int,
        prefer_low_degree: bool = True,
        seed_pid: Optional[int] = None,
        prefer_same_group: bool = True,
        avoid_products: Optional[Set[int]] = None,
    ) -> List[int]:
        avoid_products = avoid_products or set()

        candidates = [
            p for p in self._candidate_products(seed_pid, prefer_same_group)
            if p not in avoid_products
        ]

        if not candidates:
            candidates = [p for p in self.products if p not in avoid_products]

        chosen: List[int] = []
        pool = list(candidates)

        while pool and len(chosen) < k:
            weights = []
            for pid in pool:
                deg = len(self.product_reviews.get(pid, []))
                if prefer_low_degree:
                    w = 1.0 / ((deg + 1.0) ** 0.75)
                else:
                    w = (deg + 1.0) ** 0.50
                weights.append(w)

            pid = self.rng.choices(pool, weights=weights, k=1)[0]
            chosen.append(pid)
            pool.remove(pid)

        return chosen

    def _can_user_review_product(self, uid: int, pid: int) -> bool:
        return pid not in self.user_products.get(uid, set())

    def _add_review(self, uid: int, pid: int, rating: float, count_as_pattern_activity: bool = True) -> bool:
        if not self._can_user_review_product(uid, pid):
            return False

        rid_hint = len(self.kg.nodes_by_label.get("Review", [])) + 1

        # No fraud fields added to reviews.
        review_props = {
            "helpful_vote": 0,
            "images_count": 0,
            "rating": float(self._clip_star(rating)),
            "review_id": f"review_{rid_hint}_{self.rng.randint(1000, 9999)}",
            "text": "",
            "title": "",
            "verified_purchase": bool(self.rng.random() < 0.50),
        }

        rid = self.kg.add_node("Review", review_props)
        ok1 = self.kg.add_edge("WROTE", uid, rid, {}, no_duplicates=True)
        ok2 = self.kg.add_edge("ABOUT", rid, pid, {}, no_duplicates=True)

        if not (ok1 and ok2):
            return False

        self.review_to_user[rid] = uid
        self.review_to_product[rid] = pid

        self.user_reviews[uid].add(rid)
        self.user_products[uid].add(pid)

        self.product_reviews[pid].add(rid)
        self.product_users[pid].add(uid)

        self.product_ratings[pid].append(float(review_props["rating"]))
        self.user_ratings[uid].append(float(review_props["rating"]))

        if count_as_pattern_activity:
            self.injected_reviews += 1
            self.user_fraud_counts[uid] += 1
        else:
            self.injected_camouflage_reviews += 1

        return True

    def _inject_group_same_product(self, remaining: int) -> int:
        if remaining <= 0:
            return 0

        group_size = min(
            remaining,
            self.rng.randint(self.cfg.min_group_users, self.cfg.max_group_users)
        )
        users = self._weighted_pick_from_pool(group_size)
        if not users:
            return 0

        mode = self._choose_mode()
        products = self._pick_products(1, prefer_low_degree=True)
        if not products:
            return 0

        pid = products[0]
        added = 0

        for uid in users:
            if self._can_user_review_product(uid, pid):
                rating = self._pick_extreme_rating(mode)
                if self._add_review(uid, pid, rating, count_as_pattern_activity=True):
                    added += 1

        return added

    def _inject_single_user_many_products(self, remaining: int) -> int:
        if remaining <= 0:
            return 0

        users = self._weighted_pick_from_pool(1)
        if not users:
            return 0

        uid = users[0]
        mode = self._choose_mode()
        base_star = self.user_signature_star.get(uid, self._pick_extreme_rating(mode))

        k = min(
            remaining,
            self.rng.randint(self.cfg.min_burst_products, self.cfg.max_burst_products)
        )

        seed = self._pick_products(1, prefer_low_degree=True)
        if not seed:
            return 0

        reviewed = set(self.user_products.get(uid, set()))
        products = self._pick_products(
            k=k,
            prefer_low_degree=True,
            seed_pid=seed[0],
            prefer_same_group=True,
            avoid_products=reviewed,
        )

        added = 0
        for pid in products:
            if self._add_review(uid, pid, base_star, count_as_pattern_activity=True):
                added += 1

        return added

    def _inject_dense_bipartite(self, remaining: int) -> int:
        if remaining <= 0:
            return 0

        nu = self.rng.randint(self.cfg.min_dense_users, self.cfg.max_dense_users)
        np = self.rng.randint(self.cfg.min_dense_products, self.cfg.max_dense_products)

        users = self._weighted_pick_from_pool(nu)
        if not users:
            return 0

        seed = self._pick_products(1, prefer_low_degree=True)
        if not seed:
            return 0

        products = self._pick_products(
            k=np,
            prefer_low_degree=True,
            seed_pid=seed[0],
            prefer_same_group=True,
        )
        if not products:
            return 0

        density = self.rng.uniform(self.cfg.dense_edge_prob_min, self.cfg.dense_edge_prob_max)
        mode = self._choose_mode()
        star = self._pick_extreme_rating(mode)

        pairs: List[Tuple[int, int]] = []
        for uid in users:
            for pid in products:
                if self._can_user_review_product(uid, pid):
                    pairs.append((uid, pid))

        self.rng.shuffle(pairs)

        added = 0
        for uid, pid in pairs:
            if added >= remaining:
                break
            if self.rng.random() <= density:
                if self._add_review(uid, pid, star, count_as_pattern_activity=True):
                    added += 1

        if added == 0 and pairs:
            uid, pid = pairs[0]
            if self._add_review(uid, pid, star, count_as_pattern_activity=True):
                added += 1

        return added

    def _inject_near_duplicate_neighborhood(self, remaining: int) -> int:
        if remaining <= 0:
            return 0

        nu = self.rng.randint(self.cfg.min_dup_users, self.cfg.max_dup_users)
        base_k = self.rng.randint(self.cfg.min_dup_products, self.cfg.max_dup_products)

        users = self._weighted_pick_from_pool(nu)
        if not users:
            return 0

        seed = self._pick_products(1, prefer_low_degree=True)
        if not seed:
            return 0

        base_products = self._pick_products(
            k=base_k,
            prefer_low_degree=True,
            seed_pid=seed[0],
            prefer_same_group=True,
        )
        if not base_products:
            return 0

        mode = self._choose_mode()
        star = self._pick_extreme_rating(mode)

        added = 0
        for uid in users:
            take = max(1, int(round(0.8 * len(base_products))))
            subset = set(self.rng.sample(base_products, min(len(base_products), take)))

            if self.rng.random() < 0.5:
                extra = self._pick_products(
                    k=1,
                    prefer_low_degree=True,
                    seed_pid=seed[0],
                    prefer_same_group=True,
                    avoid_products=subset,
                )
                if extra:
                    subset.add(extra[0])

            for pid in subset:
                if added >= remaining:
                    return added
                if self._add_review(uid, pid, star, count_as_pattern_activity=True):
                    added += 1

        return added

    def _inject_repeated_behavior(self, remaining: int) -> int:
        if remaining <= 0:
            return 0

        nu = self.rng.randint(self.cfg.min_repeat_users, self.cfg.max_repeat_users)
        users = self._weighted_pick_from_pool(nu)
        if not users:
            return 0

        added = 0
        for uid in users:
            if added >= remaining:
                break

            k = min(
                remaining - added,
                self.rng.randint(self.cfg.min_repeat_products, self.cfg.max_repeat_products)
            )

            seed = self._pick_products(1, prefer_low_degree=True)
            if not seed:
                continue

            reviewed = set(self.user_products.get(uid, set()))
            products = self._pick_products(
                k=k,
                prefer_low_degree=True,
                seed_pid=seed[0],
                prefer_same_group=True,
                avoid_products=reviewed,
            )

            star = self.user_signature_star.get(uid, 5.0)

            for pid in products:
                if added >= remaining:
                    break
                if self._add_review(uid, pid, star, count_as_pattern_activity=True):
                    added += 1

        return added

    def _inject_camouflage(self, budget: int) -> int:
        added = 0
        trials = 0

        while added < budget and trials < max(500, budget * 10):
            trials += 1

            users = self._weighted_pick_from_pool(1)
            if not users:
                break
            uid = users[0]

            reviewed = set(self.user_products.get(uid, set()))
            products = self._pick_products(
                k=1,
                prefer_low_degree=False,
                avoid_products=reviewed,
            )
            if not products:
                continue

            pid = products[0]
            mean_rating = self._product_mean_rating(pid)

            rating = self._clip_star(
                mean_rating + self.rng.choice([-1.0, 0.0, 0.0, 0.0, 1.0])
            )

            if self._add_review(uid, pid, rating, count_as_pattern_activity=False):
                added += 1

        return added