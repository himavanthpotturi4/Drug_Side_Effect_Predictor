from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from ..config import OUTPUTS_DIR, PROCESSED_DIR
from ..utils import ensure_dir, save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class KnowledgeGraphBundle:
    adjacency: torch.Tensor
    num_drugs: int
    num_side_effects: int
    num_symptoms: int
    num_diseases: int


def _norm_adj(num_nodes: int, rows: np.ndarray, cols: np.ndarray) -> torch.Tensor:
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adj = adj + sp.eye(num_nodes, dtype=np.float32, format="coo")
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5, where=degree > 0).astype(np.float32)
    norm = sp.diags(d_inv_sqrt) @ adj @ sp.diags(d_inv_sqrt)
    return torch.tensor(norm.toarray(), dtype=torch.float32)


def _save_knowledge_graph_visualization(
    d_idx: pd.DataFrame,
    se_idx: pd.DataFrame,
    sy_idx: pd.DataFrame,
    di_idx: pd.DataFrame,
    e_se: pd.DataFrame,
    e_sy: pd.DataFrame,
    e_di: pd.DataFrame,
) -> None:
    ensure_dir(OUTPUTS_DIR)

    degree_parts = []
    for edge_df in (e_se, e_sy, e_di):
        if not edge_df.empty:
            degree_parts.append(edge_df["drug_idx"].value_counts())

    if not degree_parts:
        return

    total_degree = pd.concat(degree_parts, axis=1).fillna(0).sum(axis=1).sort_values(ascending=False)
    top_drug_ids = total_degree.head(12).index.astype(int).tolist()
    if not top_drug_ids:
        return

    se_lookup = dict(zip(se_idx["side_effect_idx"], se_idx["side_effect_name"]))
    sy_lookup = dict(zip(sy_idx["symptom_idx"], sy_idx["symptom_name"]))
    di_lookup = dict(zip(di_idx["disease_idx"], di_idx["disease_name"]))
    drug_lookup = dict(zip(d_idx["drug_idx"], d_idx["drug_name"]))

    graph = nx.Graph()
    node_colors = []
    color_map = {
        "drug": "#f59e0b",
        "side_effect": "#38bdf8",
        "symptom": "#a78bfa",
        "disease": "#34d399",
    }

    def add_node(node_id: str, label: str, node_type: str) -> None:
        if node_id not in graph:
            graph.add_node(node_id, label=label, node_type=node_type)

    summary_payload = {
        "title": "Knowledge Graph Sample",
        "description": "Representative high-connectivity drugs and a sample of their linked side effects, symptoms, and diseases.",
        "sampled_drugs": [],
        "node_type_counts": {},
        "graph_stats": {},
    }

    for drug_idx in top_drug_ids:
        drug_name = str(drug_lookup.get(drug_idx, f"Drug {drug_idx}")).title()
        drug_node = f"drug:{drug_idx}"
        add_node(drug_node, drug_name, "drug")

        se_neighbors = (
            e_se[e_se["drug_idx"] == drug_idx]["side_effect_idx"].head(3).astype(int).tolist()
            if not e_se.empty
            else []
        )
        sy_neighbors = (
            e_sy[e_sy["drug_idx"] == drug_idx]["symptom_idx"].head(2).astype(int).tolist()
            if not e_sy.empty
            else []
        )
        di_neighbors = (
            e_di[e_di["drug_idx"] == drug_idx]["disease_idx"].head(2).astype(int).tolist()
            if not e_di.empty
            else []
        )

        for side_idx in se_neighbors:
            node_id = f"side_effect:{side_idx}"
            add_node(node_id, str(se_lookup.get(side_idx, f"Side Effect {side_idx}")).title(), "side_effect")
            graph.add_edge(drug_node, node_id)

        for symptom_idx in sy_neighbors:
            node_id = f"symptom:{symptom_idx}"
            add_node(node_id, str(sy_lookup.get(symptom_idx, f"Symptom {symptom_idx}")).title(), "symptom")
            graph.add_edge(drug_node, node_id)

        for disease_idx in di_neighbors:
            node_id = f"disease:{disease_idx}"
            add_node(node_id, str(di_lookup.get(disease_idx, f"Disease {disease_idx}")).title(), "disease")
            graph.add_edge(drug_node, node_id)

        summary_payload["sampled_drugs"].append(
            {
                "drug_idx": int(drug_idx),
                "drug_name": drug_name,
                "sampled_side_effects": [
                    str(se_lookup.get(side_idx, f"Side Effect {side_idx}")).title() for side_idx in se_neighbors
                ],
                "sampled_symptoms": [
                    str(sy_lookup.get(symptom_idx, f"Symptom {symptom_idx}")).title() for symptom_idx in sy_neighbors
                ],
                "sampled_diseases": [
                    str(di_lookup.get(disease_idx, f"Disease {disease_idx}")).title() for disease_idx in di_neighbors
                ],
            }
        )

    if graph.number_of_nodes() == 0:
        return

    for _, attrs in graph.nodes(data=True):
        node_colors.append(color_map[attrs["node_type"]])

    type_counts: dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        node_type = attrs["node_type"]
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    summary_payload["node_type_counts"] = type_counts
    summary_payload["graph_stats"] = {
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "num_sampled_drugs": int(len(top_drug_ids)),
    }

    positions = nx.spring_layout(graph, seed=42, k=0.9 / np.sqrt(max(graph.number_of_nodes(), 1)), iterations=100)
    labels = {node_id: attrs["label"] for node_id, attrs in graph.nodes(data=True)}

    fig, ax = plt.subplots(figsize=(20, 15))
    nx.draw_networkx_edges(graph, positions, alpha=0.22, width=0.8, edge_color="#64748b", ax=ax)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=700,
        edgecolors="white",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8.5, font_weight="bold", ax=ax)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name.replace("_", " ").title(), markerfacecolor=color, markersize=10)
        for name, color in color_map.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, title="Node Type")
    ax.set_title("Knowledge Graph Sample", fontsize=22, fontweight="bold")
    fig.text(
        0.02,
        0.02,
        "Sample shows high-connectivity drugs with a subset of linked side effects, symptoms, and diseases.",
        fontsize=11,
        color="#334155",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "knowledge_graph_sample.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUTS_DIR / "knowledge_graph_sample_presentation.png", dpi=320, bbox_inches="tight")
    plt.close(fig)
    save_json(OUTPUTS_DIR / "knowledge_graph_sample_summary.json", summary_payload)


def run_module_3_knowledge_graph_construction(export_visualization: bool = True) -> KnowledgeGraphBundle:
    d_idx = pd.read_csv(PROCESSED_DIR / "drug_index.csv")
    se_idx = pd.read_csv(PROCESSED_DIR / "side_effect_index.csv")
    sy_idx = pd.read_csv(PROCESSED_DIR / "symptom_index.csv")
    di_idx = pd.read_csv(PROCESSED_DIR / "disease_index.csv")

    e_se = pd.read_csv(PROCESSED_DIR / "drug_side_effect_edges.csv")
    e_sy = pd.read_csv(PROCESSED_DIR / "drug_symptom_edges.csv")
    e_di = pd.read_csv(PROCESSED_DIR / "drug_disease_edges.csv")

    num_drugs = len(d_idx)
    num_side = len(se_idx)
    num_symp = len(sy_idx)
    num_dis = len(di_idx)

    side_offset = num_drugs
    symp_offset = num_drugs + num_side
    dis_offset = num_drugs + num_side + num_symp
    num_nodes = num_drugs + num_side + num_symp + num_dis

    rows = []
    cols = []

    if not e_se.empty:
        r = e_se["drug_idx"].to_numpy()
        c = e_se["side_effect_idx"].to_numpy() + side_offset
        rows.extend(np.concatenate([r, c]))
        cols.extend(np.concatenate([c, r]))
    if not e_sy.empty:
        r = e_sy["drug_idx"].to_numpy()
        c = e_sy["symptom_idx"].to_numpy() + symp_offset
        rows.extend(np.concatenate([r, c]))
        cols.extend(np.concatenate([c, r]))
    if not e_di.empty:
        r = e_di["drug_idx"].to_numpy()
        c = e_di["disease_idx"].to_numpy() + dis_offset
        rows.extend(np.concatenate([r, c]))
        cols.extend(np.concatenate([c, r]))

    if export_visualization:
        _save_knowledge_graph_visualization(d_idx, se_idx, sy_idx, di_idx, e_se, e_sy, e_di)

    adjacency = _norm_adj(num_nodes, np.asarray(rows), np.asarray(cols))
    return KnowledgeGraphBundle(
        adjacency=adjacency,
        num_drugs=num_drugs,
        num_side_effects=num_side,
        num_symptoms=num_symp,
        num_diseases=num_dis,
    )
