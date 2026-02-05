import string
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import random
import matplotlib.pyplot as plt
import networkx as nx

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

def visualize_matplotlib(kg, max_nodes=200, max_edges=500, seed=42):
    rng = random.Random(seed)

    all_edges = []
    for rel_type, edges in kg.edges_by_type.items():
        for e in edges:
            all_edges.append((e.src, e.dst, rel_type))

    node_ids = [n.id for n in kg.nodes]
    if len(node_ids) > max_nodes:
        keep = set(rng.sample(node_ids, max_nodes))
    else:
        keep = set(node_ids)

    if len(all_edges) > max_edges:
        all_edges = rng.sample(all_edges, max_edges)

    G = nx.DiGraph()
    for nid in keep:
        label = kg.node_label.get(nid, "Node")
        props = kg.node_props.get(nid, {})
        name = props.get("name") or props.get("product_name") or props.get("email") or ""
        G.add_node(nid, label=label, name=name)

    for s, t, r in all_edges:
        if s in keep and t in keep:
            G.add_edge(s, t, rel=r)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=seed, k=0.35)

    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_size=300)

    labels = {}
    for nid, data in G.nodes(data=True):
        base = data["label"]
        if data.get("name"):
            base += f"\n{data['name']}"
        labels[nid] = base

    nx.draw_networkx_labels(G, pos, labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
