from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from ..config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR
from ..utils import ensure_dir, save_json
from .module_3_knowledge_graph_construction import run_module_3_knowledge_graph_construction
from .module_4_graph_neural_network import ArchitectureGNN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_splits(num_drugs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_ids = np.arange(num_drugs)
    train_ids, test_ids = train_test_split(all_ids, test_size=0.15, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)
    return train_ids, val_ids, test_ids


def _save_confusion_matrix_plot(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path,
) -> None:
    fig_width = max(6, min(14, 1.1 * len(labels) + 2))
    fig_height = max(5, min(12, 0.9 * len(labels) + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            color = "white" if value > threshold else "#0f172a"
            ax.text(j, i, value, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_embedding_projection_plot(embedding_df: pd.DataFrame, output_path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    classes = sorted(embedding_df["predicted_target_class"].dropna().unique().tolist())
    cmap = plt.cm.get_cmap("tab10", max(len(classes), 1))

    for idx, class_name in enumerate(classes):
        subset = embedding_df[embedding_df["predicted_target_class"] == class_name]
        ax.scatter(
            subset["x"],
            subset["y"],
            s=24,
            alpha=0.72,
            label=class_name.replace("_", " ").title(),
            color=cmap(idx),
            edgecolors="none",
        )

    ax.set_title("Drug Node Embeddings - 2D PCA Projection")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    if classes:
        ax.legend(title="Predicted Class", frameon=True, fontsize=9)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_metrics_bar_chart(metrics: dict, output_path) -> None:
    metric_labels = {
        "action_accuracy": "Action Accuracy",
        "action_auc_ovr_macro": "Action AUC (Macro)",
        "action_f1_macro": "Action F1 (Macro)",
        "action_f1_weighted": "Action F1 (Weighted)",
        "side_effect_f1_micro": "Side-Effect F1 (Micro)",
    }
    chart_rows = [
        {"label": metric_labels[key], "value": float(metrics[key])}
        for key in metric_labels
        if metrics.get(key) is not None
    ]
    if not chart_rows:
        return

    labels = [row["label"] for row in chart_rows]
    values = [row["value"] for row in chart_rows]
    colors = ["#2563eb", "#0ea5e9", "#14b8a6", "#22c55e", "#f59e0b"][: len(values)]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics", fontsize=18, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 0.98),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_module_5_model_training(epochs: int = 50, lr: float = 1e-3) -> dict:
    ensure_dir(MODELS_DIR)
    ensure_dir(OUTPUTS_DIR)

    kg = run_module_3_knowledge_graph_construction()
    edges = pd.read_csv(PROCESSED_DIR / "drug_side_effect_edges.csv")
    action_labels = pd.read_csv(PROCESSED_DIR / "drug_action_labels.csv")
    action_index = pd.read_csv(PROCESSED_DIR / "action_index.csv")
    drug_index = pd.read_csv(PROCESSED_DIR / "drug_index.csv")

    y_side = np.zeros((kg.num_drugs, kg.num_side_effects), dtype=np.float32)
    y_side[edges["drug_idx"].to_numpy(), edges["side_effect_idx"].to_numpy()] = 1.0
    y_action = np.zeros(kg.num_drugs, dtype=np.int64)
    y_action[action_labels["drug_idx"].to_numpy()] = action_labels["action_idx"].to_numpy()

    train_ids, val_ids, test_ids = _build_splits(kg.num_drugs)
    train_mask = torch.zeros(kg.num_drugs, dtype=torch.bool)
    val_mask = torch.zeros(kg.num_drugs, dtype=torch.bool)
    test_mask = torch.zeros(kg.num_drugs, dtype=torch.bool)
    train_mask[train_ids] = True
    val_mask[val_ids] = True
    test_mask[test_ids] = True

    model = ArchitectureGNN(
        num_nodes=kg.adjacency.shape[0],
        num_drugs=kg.num_drugs,
        num_side_effects=kg.num_side_effects,
        num_actions=len(action_index),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    class_counts = np.bincount(y_action[train_ids], minlength=len(action_index))
    class_counts[class_counts == 0] = 1
    class_weights = torch.tensor((class_counts.sum() / class_counts), dtype=torch.float32)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    y_side_t = torch.tensor(y_side, dtype=torch.float32)
    y_action_t = torch.tensor(y_action, dtype=torch.long)
    adj = kg.adjacency

    history = []
    best_state = None
    best_val = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        side_logits, action_logits, _ = model(adj)
        loss = bce(side_logits[train_mask], y_side_t[train_mask]) + 0.7 * ce(
            action_logits[train_mask], y_action_t[train_mask]
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            side_logits, action_logits, _ = model(adj)
            val_pred = torch.argmax(action_logits[val_mask], dim=1)
            val_true = y_action_t[val_mask]
            val_acc = (val_pred == val_true).float().mean().item()
            history.append({"epoch": epoch, "loss": float(loss.item()), "val_action_acc": float(val_acc)})
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        side_logits, action_logits, embeddings = model(adj)
        test_pred = torch.argmax(action_logits[test_mask], dim=1).cpu().numpy()
        test_true = y_action_t[test_mask].cpu().numpy()
        test_probs = torch.softmax(action_logits[test_mask], dim=1).cpu().numpy()

        side_test_true = y_side_t[test_mask].cpu().numpy()
        side_test_prob = torch.sigmoid(side_logits[test_mask]).cpu().numpy()
        side_test_pred = (side_test_prob >= 0.35).astype(np.int32)

    action_acc = float(accuracy_score(test_true, test_pred))
    action_f1_macro = float(f1_score(test_true, test_pred, average="macro", zero_division=0))
    action_f1_weighted = float(f1_score(test_true, test_pred, average="weighted", zero_division=0))
    side_f1_micro = float(f1_score(side_test_true, side_test_pred, average="micro", zero_division=0))

    try:
        y_true_ovr = np.zeros((len(test_true), len(action_index)), dtype=np.float32)
        y_true_ovr[np.arange(len(test_true)), test_true] = 1.0
        action_auc_ovr = float(roc_auc_score(y_true_ovr, test_probs, average="macro", multi_class="ovr"))
    except Exception:
        action_auc_ovr = None

    cm = confusion_matrix(test_true, test_pred, labels=list(range(len(action_index))))
    action_labels_pretty = [x.replace("_", " ").title() for x in action_index["action_class"].tolist()]
    cm_df = pd.DataFrame(cm, index=action_labels_pretty, columns=action_labels_pretty)
    cm_df.to_csv(OUTPUTS_DIR / "confusion_matrix_action.csv")
    _save_confusion_matrix_plot(
        cm,
        action_labels_pretty,
        "Confusion Matrix - Action Classification",
        OUTPUTS_DIR / "confusion_matrix_action.png",
    )

    # Binary confusion matrix for side-effect link prediction (aggregated over all test pairs).
    y_true_flat = side_test_true.reshape(-1).astype(np.int32)
    y_pred_flat = side_test_pred.reshape(-1).astype(np.int32)
    bin_cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
    tn, fp, fn, tp = int(bin_cm[0, 0]), int(bin_cm[0, 1]), int(bin_cm[1, 0]), int(bin_cm[1, 1])
    bin_cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )
    bin_cm_df.to_csv(OUTPUTS_DIR / "confusion_matrix_side_effect_binary.csv")
    _save_confusion_matrix_plot(
        np.array([[tn, fp], [fn, tp]]),
        ["Predicted Negative", "Predicted Positive"],
        "Confusion Matrix - Side-Effect Prediction (Binary)",
        OUTPUTS_DIR / "confusion_matrix_side_effect_binary.png",
    )
    save_json(
        OUTPUTS_DIR / "confusion_matrix_side_effect_binary_summary.json",
        {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
    )

    # Embedding visualization (2D PCA of drug embeddings).
    drug_embeddings = embeddings[: kg.num_drugs].detach().cpu().numpy()
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(drug_embeddings)
    emb_df = pd.DataFrame(
        {
            "drug_idx": np.arange(kg.num_drugs),
            "drug_name": drug_index["drug_name"].astype(str),
            "x": xy[:, 0],
            "y": xy[:, 1],
        }
    )
    full_pred = torch.argmax(action_logits, dim=1).detach().cpu().numpy()
    class_map = dict(zip(action_index["action_idx"], action_index["action_class"]))
    emb_df["predicted_target_class"] = [class_map.get(int(i), "unknown") for i in full_pred]
    emb_df.to_csv(OUTPUTS_DIR / "embedding_projection_drugs.csv", index=False)
    _save_embedding_projection_plot(emb_df, OUTPUTS_DIR / "embedding_projection_drugs.png")

    checkpoint = {
        "state_dict": model.state_dict(),
        "num_nodes": kg.adjacency.shape[0],
        "num_drugs": kg.num_drugs,
        "num_side_effects": kg.num_side_effects,
        "num_actions": len(action_index),
        "threshold": 0.35,
    }
    torch.save(checkpoint, MODELS_DIR / "architecture_gnn.pt")
    pd.DataFrame(history).to_csv(OUTPUTS_DIR / "architecture_training_history.csv", index=False)
    metrics = {
        "action_accuracy": action_acc,
        "action_auc_ovr_macro": action_auc_ovr,
        "action_f1_macro": action_f1_macro,
        "action_f1_weighted": action_f1_weighted,
        "side_effect_f1_micro": side_f1_micro,
        "num_test_samples": int(len(test_true)),
    }
    save_json(OUTPUTS_DIR / "architecture_test_metrics.json", metrics)
    _save_metrics_bar_chart(metrics, OUTPUTS_DIR / "architecture_test_metrics.png")
    return {**metrics, "epochs": epochs}
