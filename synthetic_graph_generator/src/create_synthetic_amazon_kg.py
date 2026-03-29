from datasets import load_dataset
from kg_generator import generate_kg_from_schema
import json
from schema_generator import save_inferred_schema
from typing import Dict
from utils import build_amazon_reviews_kg, save_kg_to_neo4j, visualize_full_kg
from review_fraud_injector import FraudInjectionConfig, ReviewKGUserFraudInjector
from review_pattern_sanitizer import SanitizeConfig, ReviewKGPatternSanitizer


def main():
    # kg = build_amazon_reviews_kg("raw_review_Digital_Music")

    # print("Total nodes:", len(kg.nodes))
    # for label, ids in kg.nodes_by_label.items():
    #     print(f"{label}: {len(ids)}")

    # print("\nEdges:")
    # for rel_type, edges in kg.edges_by_type.items():
    #     print(f"{rel_type}: {len(edges)}")

    # schema = save_inferred_schema(kg, "synthetic_graph_generator/schemas/amazon_inferred_schema.json")
    # print(json.dumps(schema, indent=2))

    schema = "synthetic_graph_generator/schemas/amazon_inferred_schema.json"

    kg = generate_kg_from_schema(schema)

    sanitize_cfg = SanitizeConfig(
        seed=42,
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

    sanitizer = ReviewKGPatternSanitizer(kg, sanitize_cfg)
    kg, sanitize_stats = sanitizer.sanitize()
    print("SANITIZE:", sanitize_stats)

    fraud_cfg = FraudInjectionConfig(
        seed=42,
        corruption_rate=0.05,
        criminal_user_fraction=0.01,
        min_criminal_users=200,
        camouflage_rate=0.30,
    )

    injector = ReviewKGUserFraudInjector(kg, fraud_cfg)
    inject_stats = injector.inject()
    print("INJECT:", inject_stats)

    save_kg_to_neo4j(
        kg,
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="neo4j"
    )

if __name__ == "__main__":
    main()