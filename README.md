# Synthetic Knowledge Graph Framework for Fraud Detection

This repository contains an end-to-end framework for generating synthetic knowledge graphs for fraud detection in the Amazon review domain. The project addresses a common limitation in graph-based fraud research: realistic fraud graphs are often hard to access because of privacy restrictions, proprietary constraints, labeling cost, and general data scarcity. The framework builds a clean synthetic review graph from an inferred schema, evaluates its structural fidelity against the original graph, sanitizes suspicious pre-existing patterns, injects controlled fraud motifs, and trains a heterogeneous GNN for user-level fraud detection.

## Project Goals

The main goals of this project are:
- generate schema-driven synthetic knowledge graphs that preserve typed entities, typed relations, and structural constraints,
- inject controlled user-level fraud patterns into the generated graphs,
- compare original and synthetic graphs using structural fidelity metrics,
- evaluate downstream fraud detection with a heterogeneous GraphSAGE model.

## Domain

The current implementation is built around the Amazon Reviews 2023 dataset. The graph is heterogeneous and includes the node types `User`, `Review`, `Product`, and `ProductGroup`, connected by the relations `WROTE`, `ABOUT`, and `BELONGS_TO`. The project treats fraud as a user-level prediction problem and uses structural reviewer-product motifs to model suspicious behavior.

## Repository Structure

### Core files

- `utils.py`  
  Defines the in-memory KG data structures, utilities for sampling and power-law degree generation, Amazon review graph construction, and Neo4j export helpers.

- `schema_generator.py`  
  Infers node and relationship schema definitions directly from a knowledge graph, including property types, counts, and degree constraints.

- `schema_text_parser.py`  
  Parses the pseudo-graph schema format into the internal schema representation used by the generator.

- `kg_generator.py`  
  Generates a synthetic KG from a schema. It creates nodes, allocates relation degrees, and materializes edges while respecting relationship constraints such as degree bounds and duplicate-edge rules.

- `review_pattern_sanitizer.py`  
  Sanitizes a clean synthetic graph by detecting and removing users that already exhibit suspicious rating or interaction behavior before controlled fraud injection.

- `review_fraud_injector.py`  
  Injects user-level fraud motifs into the synthetic graph using configurable corruption settings, criminal-pool selection, pattern weights, reuse bias, and camouflage behavior.

- `evaluation_kg.py`  
  Evaluates structural fidelity between original and synthetic graphs using node counts, edge counts, typed degree distributions, Jensen–Shannon divergence, and chi-square goodness-of-fit summaries.

- `gnn.py`  
  Builds heterogeneous graph features and trains a heterogeneous GraphSAGE model for fraud detection. Supports both single-split training and k-fold evaluation.

- `run_report_pipeline.py`  
  Runs the full pipeline across small, medium, and large dataset configurations, writes summaries to disk, and reports synthetic-test and transfer metrics.

- `A Framework for Synthetic Knowledge Graph.pdf`  
  Project report describing the motivation, methodology, evaluation design, results, and conclusions.

## End-to-End Pipeline

The overall workflow is:

1. Build the original Amazon review KG.
2. Infer a schema from the original graph.
3. Generate a clean synthetic KG from that schema.
4. Evaluate structural fidelity between the original and synthetic graphs.
5. Sanitize suspicious naturally occurring patterns in the clean synthetic graph.
6. Inject controlled user-level fraud motifs.
7. Convert the corrupted graph into a heterogeneous learning graph.
8. Train and evaluate a heterogeneous GraphSAGE fraud detector. 

## Fraud Motifs

The fraud injector supports multiple configurable motif families, including:
- same-product coordination,
- single-user many-products behavior,
- dense bipartite reviewer-product blocks,
- near-duplicate neighborhoods,
- repeated behavior patterns.

## Features

- Schema-driven synthetic KG generation
- Power-law-inspired degree allocation
- Structural fidelity evaluation
- Pre-injection sanitization
- Controlled fraud motif injection
- Heterogeneous GraphSAGE training
- Single split and k-fold evaluation
- Optional Neo4j export for graph inspection and storage

## Requirements

This project uses Python and depends on packages imported by the repository, including:
- `torch`
- `torch_geometric`
- `datasets`
- `neo4j`
- `matplotlib`
- `networkx`
- `scipy`  
Some scripts also rely on standard-library modules such as `argparse`, `json`, `pickle`, and `dataclasses`.
