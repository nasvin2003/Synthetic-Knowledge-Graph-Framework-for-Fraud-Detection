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
    ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", category, split="full")

    g = KG()

    user_to_nid: Dict[str, int] = {}
    product_to_nid: Dict[str, int] = {}
    group_to_nid: Dict[str, int] = {}

    product_group_assigned: Dict[int, int] = {}

    add_node = g.add_node
    add_edge = g.add_edge

    for i, row in enumerate(ds):
        user_id = row.get("user_id")
        asin = row.get("asin")
        parent_asin = row.get("parent_asin")

        if not user_id or not asin:
            continue

        user_nid = user_to_nid.get(user_id)
        if user_nid is None:
            user_nid = add_node("User", {"user_id": user_id})
            user_to_nid[user_id] = user_nid

        product_nid = product_to_nid.get(asin)
        if product_nid is None:
            product_nid = add_node("Product", {"asin": asin})
            product_to_nid[asin] = product_nid

        title = row.get("title")
        if title is None:
            title = ""

        text = row.get("text")
        if text is None:
            text = ""

        helpful_vote = row.get("helpful_vote")
        if helpful_vote is None:
            helpful_vote = 0

        verified_purchase = row.get("verified_purchase")
        if verified_purchase is None:
            verified_purchase = False

        rating = row.get("rating")

        images = row.get("images")
        images_count = len(images) if isinstance(images, list) else 0

        review_nid = add_node("Review", {
            "review_id": f"review_{i}",
            "rating": rating,
            "title": title,
            "text": text,
            "helpful_vote": helpful_vote,
            "verified_purchase": bool(verified_purchase),
            "images_count": images_count,
        })

        add_edge("WROTE", user_nid, review_nid, {}, no_duplicates=False)
        add_edge("ABOUT", review_nid, product_nid, {}, no_duplicates=False)

        if parent_asin is not None:
            parent_asin = str(parent_asin).strip()
            if parent_asin:
                group_nid = group_to_nid.get(parent_asin)
                if group_nid is None:
                    group_nid = add_node("ProductGroup", {"parent_asin": parent_asin})
                    group_to_nid[parent_asin] = group_nid

                assigned_gid = product_group_assigned.get(product_nid)
                if assigned_gid is None:
                    add_edge("BELONGS_TO", product_nid, group_nid, {}, no_duplicates=False)
                    product_group_assigned[product_nid] = group_nid

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
            session.run("MATCH (n) DETACH DELETE n")

            print("Inserting nodes into Neo4j...")
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
