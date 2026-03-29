from collections import defaultdict
import string
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import random
import matplotlib.pyplot as plt
import networkx as nx
from datasets import load_dataset
from dataclasses import dataclass
import matplotlib.patches as mpatches
import math

import neo4j

@dataclass
class Node:
    id: int
    label: str
    props: Dict[str, Any]

@dataclass
class Edge:
    src: int
    dst: int
    rel_type: str
    props: Dict[str, Any]

class KG:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self.nodes_by_label: Dict[str, List[int]] = defaultdict(list)
        self.node_props: Dict[int, Dict[str, Any]] = {}
        self.node_label: Dict[int, str] = {}
        self.edges_by_type: Dict[str, List[Edge]] = defaultdict(list)
        self.edge_keys: Dict[str, set] = defaultdict(set)

    def add_node(self, label: str, props: Dict[str, Any]) -> int:
        nid = len(self.nodes) + 1
        props2 = dict(props)
        props2["id"] = nid
        self.nodes.append(Node(id=nid, label=label, props=props2))
        self.nodes_by_label[label].append(nid)
        self.node_props[nid] = props2
        self.node_label[nid] = label
        return nid

    def add_edge(self, rel_type: str, src: int, dst: int, props: Dict[str, Any], no_duplicates: bool = True) -> bool:
        key = (src << 32) | dst
        if no_duplicates:
            if key in self.edge_keys[rel_type]:
                return False
        self.edge_keys[rel_type].add(key)
        self.edges_by_type[rel_type].append(Edge(src=src, dst=dst, rel_type=rel_type, props=dict(props)))
        return True

def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x

def sample_int_uniform(lo: int, hi: int, rng: random.Random) -> int:
    return rng.randint(lo, hi)

def sample_str(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(12))

def powerlaw_weights(kmin: int, kmax: int, alpha: float) -> List[float]:
    return [((k + 1.0) ** (-alpha)) for k in range(kmin, kmax + 1)]

def cdf_from_weights(weights: List[float]) -> List[float]:
    s = sum(weights)
    cdf: List[float] = []
    acc = 0.0
    for w in weights:
        acc += w / s
        cdf.append(acc)
    cdf[-1] = 1.0
    return cdf

def sample_from_cdf(cdf: List[float], rng: random.Random) -> int:
    u = rng.random()
    lo, hi = 0, len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] >= u:
            hi = mid
        else:
            lo = mid + 1
    return lo

from typing import Optional
from neo4j import GraphDatabase

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def build_amazon_reviews_kg(category: str = "raw_review_Digital_Music") -> KG:
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", category)
    ds = dataset["full"]

    g = KG()

    user_to_nid: Dict[str, int] = {}
    product_to_nid: Dict[str, int] = {}
    group_to_nid: Dict[str, int] = {}

    # optional: avoid duplicate BELONGS_TO creation attempts
    seen_product_group_pairs = set()

    def get_user_node(user_id: str) -> int:
        nid = user_to_nid.get(user_id)
        if nid is None:
            nid = g.add_node("User", {
                "user_id": user_id
            })
            user_to_nid[user_id] = nid
        return nid

    def get_product_node(asin: str) -> int:
        nid = product_to_nid.get(asin)
        if nid is None:
            nid = g.add_node("Product", {
                "asin": asin
            })
            product_to_nid[asin] = nid
        return nid

    def get_group_node(parent_asin: str) -> int:
        nid = group_to_nid.get(parent_asin)
        if nid is None:
            nid = g.add_node("ProductGroup", {
                "parent_asin": parent_asin
            })
            group_to_nid[parent_asin] = nid
        return nid

    for i, row in enumerate(ds):
        user_id = row.get("user_id")
        asin = row.get("asin")
        parent_asin = row.get("parent_asin")

        if not user_id or not asin:
            continue

        title = row.get("title")
        text = row.get("text")
        helpful_vote = row.get("helpful_vote")
        verified_purchase = row.get("verified_purchase")
        rating = row.get("rating")
        images = row.get("images")

        if title is None:
            title = ""
        if text is None:
            text = ""
        if helpful_vote is None:
            helpful_vote = 0
        if verified_purchase is None:
            verified_purchase = False

        images_count = len(images) if isinstance(images, list) else 0

        user_nid = get_user_node(user_id)
        product_nid = get_product_node(asin)

        review_nid = g.add_node("Review", {
            "review_id": f"review_{i}",
            "rating": rating,
            "title": title,
            "text": text,
            "helpful_vote": helpful_vote,
            "verified_purchase": bool(verified_purchase),
            "images_count": images_count
        })

        g.add_edge("WROTE", user_nid, review_nid, {}, no_duplicates=True)
        g.add_edge("ABOUT", review_nid, product_nid, {}, no_duplicates=True)

        if parent_asin is not None and str(parent_asin).strip() != "":
            group_nid = get_group_node(str(parent_asin))

            pair_key = (product_nid, group_nid)
            if pair_key not in seen_product_group_pairs:
                g.add_edge("BELONGS_TO", product_nid, group_nid, {}, no_duplicates=True)
                seen_product_group_pairs.add(pair_key)

    return g

def save_kg_to_neo4j(
    kg,
    uri: str,
    user: str,
    password: str,
    database: Optional[str] = None,
    batch_size: int = 10000
):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session(database=database) as session:
            # Clear database first
            # session.run("USE "+database+" RETURN 1") if database else None
            session.run("MATCH (n) DETACH DELETE n")

            print("Inserting nodes into Neo4j...")
            # -------------------------
            # Batch insert nodes
            # -------------------------
            nodes_by_label = {}

            for node in kg.nodes:
                nid = node.id
                label = kg.node_label.get(nid, "Node")
                props = dict(kg.node_props.get(nid, {}))
                props["id"] = nid

                if label not in nodes_by_label:
                    nodes_by_label[label] = []
                nodes_by_label[label].append(props)

            for label, rows in nodes_by_label.items():
                query = f"""
                UNWIND $rows AS row
                CREATE (n:{label})
                SET n = row
                """

                for batch in chunked(rows, batch_size):
                    session.run(query, rows=batch)

                session.run("CREATE INDEX nid_" + label + " IF NOT EXISTS FOR (n:" + label + ") ON (n.id)")

            # -------------------------
            # Batch insert edges
            # -------------------------
            print("Inserting edges into Neo4j...")

            grouped_edges = defaultdict(list)

            for rel_type, edges in kg.edges_by_type.items():
                for edge in edges:
                    from_label = kg.node_label[edge.src]
                    to_label = kg.node_label[edge.dst]
                    grouped_edges[(rel_type, from_label, to_label)].append({
                        "from_id": edge.src,
                        "to_id": edge.dst,
                        "props": dict(edge.props)
                    })

            for (rel_type, from_label, to_label), rows in grouped_edges.items():
                query = f"""
                UNWIND $rows AS row
                MATCH (a:{from_label} {{id: row.from_id}})
                MATCH (b:{to_label} {{id: row.to_id}})
                CREATE (a)-[r:{rel_type}]->(b)
                SET r = row.props
                """

                total_created = 0
                for batch in chunked(rows, batch_size):
                    result = session.run(query, rows=batch)
                    summary = result.consume()
                    total_created += summary.counters.relationships_created
    finally:
        driver.close()

def visualize_full_kg(
    kg,
    figsize: Tuple[int, int] = (24, 24),
    seed: int = 42,
    force: bool = False,
    show_edge_colors: bool = False,
    name: str = "kg_visualization.png"
):
    """
    Visualize the entire KG.

    Rules:
    - Node colors are based on node label.
    - If a node has props["fraud"] == True, it is always drawn in red.
    - Red is reserved only for fraud nodes. Non-fraud nodes never use red.
    """

    num_nodes = len(kg.nodes)
    num_edges = sum(len(edges) for edges in kg.edges_by_type.values())

    if not force and num_nodes > 15000:
        raise ValueError(
            f"Graph is very large ({num_nodes} nodes, {num_edges} edges). "
            f"Rendering the entire KG may be very slow. "
            f"Use force=True if you still want to draw it."
        )

    G = nx.DiGraph()

    for node in kg.nodes:
        props = kg.node_props.get(node.id, {})
        is_fraud = bool(props.get("fraud", False))
        G.add_node(node.id, label=node.label, fraud=is_fraud)

    rel_types_sorted = sorted(kg.edges_by_type.keys())

    for rel_type, edges in kg.edges_by_type.items():
        for e in edges:
            G.add_edge(e.src, e.dst, rel_type=rel_type)

    node_labels_sorted = sorted(kg.nodes_by_label.keys())

    # No red in this palette
    base_node_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#393b79",
        "#637939",
        "#8c6d31",
        "#7b4173",
        "#3182bd",
        "#31a354",
    ]

    fraud_color = "#ff0000"

    label_to_color: Dict[str, str] = {}
    for i, label in enumerate(node_labels_sorted):
        label_to_color[label] = base_node_colors[i % len(base_node_colors)]

    base_edge_colors = [
        "#aaaaaa", "#555555", "#9999cc", "#cc9999", "#99cc99",
        "#c7c7c7", "#8da0cb", "#fc8d62", "#66c2a5"
    ]
    rel_to_color: Dict[str, str] = {}
    for i, rel in enumerate(rel_types_sorted):
        rel_to_color[rel] = base_edge_colors[i % len(base_edge_colors)]

    if num_nodes <= 2000:
        pos = nx.spring_layout(G, seed=seed, k=None, iterations=50)
    elif num_nodes <= 10000:
        pos = nx.spring_layout(
            G,
            seed=seed,
            k=1.0 / math.sqrt(max(1, num_nodes)),
            iterations=20
        )
    else:
        rng = random.Random(seed)
        pos = {n: (rng.random(), rng.random()) for n in G.nodes()}

    plt.figure(figsize=figsize)

    if show_edge_colors:
        edge_colors = []
        for _, _, data in G.edges(data=True):
            edge_colors.append(rel_to_color.get(data.get("rel_type", ""), "#bbbbbb"))

        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            alpha=0.12 if num_nodes > 5000 else 0.20,
            arrows=False,
            width=0.3 if num_nodes > 5000 else 0.5,
        )
    else:
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="#999999",
            alpha=0.10 if num_nodes > 5000 else 0.18,
            arrows=False,
            width=0.25 if num_nodes > 5000 else 0.4,
        )

    if num_nodes > 100000:
        node_size = 2
    elif num_nodes > 20000:
        node_size = 4
    elif num_nodes > 5000:
        node_size = 8
    else:
        node_size = 20

    fraud_nodes: List[int] = []
    normal_nodes_by_label: Dict[str, List[int]] = {label: [] for label in node_labels_sorted}

    for node in kg.nodes:
        props = kg.node_props.get(node.id, {})
        if bool(props.get("fraud", False)):
            fraud_nodes.append(node.id)
        else:
            normal_nodes_by_label[node.label].append(node.id)

    legend_handles = []

    # Draw normal nodes first
    for label in node_labels_sorted:
        nodelist = normal_nodes_by_label[label]
        if not nodelist:
            continue

        color = label_to_color[label]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_color=color,
            node_size=node_size,
            alpha=0.85,
            linewidths=0.0,
        )

        legend_handles.append(mpatches.Patch(color=color, label=label))

    # Draw fraud nodes last so they stay visible on top
    if fraud_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=fraud_nodes,
            node_color=fraud_color,
            node_size=max(node_size * 1.4, node_size + 4),
            alpha=0.95,
            linewidths=0.2,
            edgecolors="black" if num_nodes <= 10000 else None,
        )
        legend_handles.append(mpatches.Patch(color=fraud_color, label="Fraud"))

    plt.title(f"Knowledge Graph Visualization ({num_nodes} nodes, {num_edges} edges)")
    plt.axis("off")
    plt.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=10,
        frameon=True
    )
    plt.savefig(name, dpi=300)
    plt.tight_layout()
    plt.show()