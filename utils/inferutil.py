# Import commands
import os
import random
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import operator
from torch.cuda import set_device

## 3 Main funcs for inference of a single example, inference of all examples or subset, testing different thresholds and label conditions for all examples or subset


# Helper func to calculate and get metrics
def get_metrics(pred_labels, act_labels):
    """
    Calculates metrics for prediction vs. actual values, including bold and non-bold counts for both.

    Args:
        pred_labels (Tensor): Predicted labels as a 1D flattened Tensor.
        act_labels (Tensor): Actual labels as a 1D flattened Tensor.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, F1-score, predicted and actual bold/non-bold counts,
              and the confusion matrix.
    """
    # Convert to numpy arrays for metric calculations
    pred_np = pred_labels.cpu().numpy().flatten()
    act_np = act_labels.cpu().numpy().flatten()

    # Return metrics dictionary with inline calculations
    return {
        "accuracy": accuracy_score(act_np, pred_np),
        "precision": precision_score(act_np, pred_np, pos_label=1, zero_division=0),
        "recall": recall_score(act_np, pred_np, pos_label=1, zero_division=0),
        "f1": f1_score(act_np, pred_np, pos_label=1, zero_division=0),
        "pred_bold_count": (pred_labels == 1).sum().item(),
        "pred_non_bold_count": (pred_labels == 0).sum().item(),
        "act_bold_count": (act_labels == 1).sum().item(),
        "act_non_bold_count": (act_labels == 0).sum().item(),
        "confusion_matrix": confusion_matrix(act_np, pred_np, labels=[0, 1]),
    }


def infer_one(
    trained_model,
    infer_loader,
    loc=0,
    threshold=0.5,
    condition=">",
    disp_max=False,
    device="cuda:0",
):
    # Set pandas display options
    [
        pd.set_option(opt, None) if disp_max else pd.reset_option(opt)
        for opt in ["display.max_rows", "display.max_columns"]
    ]

    # Define valid conditions
    conditions = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    if condition not in conditions:
        raise ValueError(
            f"Invalid condition '{condition}'. Must be one of {list(conditions.keys())}."
        )

    # Perform inference
    trained_model.eval()
    with torch.no_grad():
        predictions = trained_model(infer_loader.x_tok[loc].unsqueeze(0).to(device))
    pred_probs = torch.sigmoid(predictions.squeeze(0))

    # Apply the condition to compute pred_labels
    pred_labels = conditions[condition](pred_probs, threshold).long()

    # pred_labels = (pred_probs > threshold).long()
    act_labels = torch.tensor(infer_loader.y_tok[loc][:, :, 6].numpy()).to(device)
    metrics = get_metrics(pred_labels, act_labels)

    # Display results
    print(f"\nFilename: {infer_loader.file_paths[loc]}")

    # --- Unique Sigmoid Values for Bold Cells ---
    bold_indices = torch.nonzero(
        act_labels == 1, as_tuple=False
    )  # Get locations of bold cells
    print("\n--- Unique Sigmoid Probabilities for Bold Cells ---")
    if len(bold_indices) > 0:
        # Dictionary to store unique sigmoid values and one example location
        unique_sigmoids = {}
        for idx in bold_indices:
            row, col = idx.tolist()
            sigmoid_value = pred_probs[row, col].item()
            if sigmoid_value not in unique_sigmoids:
                unique_sigmoids[sigmoid_value] = (
                    row,
                    col,
                )  # Store the first occurrence

        # Sort the dictionary by sigmoid values in ascending order
        sorted_sigmoids = sorted(
            unique_sigmoids.items(), key=lambda x: x[0], reverse=False
        )

        # Print unique sigmoid values with example locations
        for value, location in sorted_sigmoids:
            row, col = location
            print(f"({row},{col}): {value:.20f}")
    else:
        print("No bold cells in the actual data.")

    print(
        f"\nNB to B ratio: Predicted = {metrics['pred_non_bold_count']}:{metrics['pred_bold_count']} | "
        f"Actual = {metrics['act_non_bold_count']}:{metrics['act_bold_count']}\n"
        f"Accuracy: {metrics['accuracy'] * 100:.2f}% | Precision: {metrics['precision'] * 100:.2f}% | "
        f"Recall: {metrics['recall'] * 100:.2f}% | F1-Score: {metrics['f1']:.2f}\n"
    )
    # Display confusion matrix
    cm = metrics["confusion_matrix"]
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=[
            ["TN\n" + str(cm[0, 0]), "FP\n" + str(cm[0, 1])],
            ["FN\n" + str(cm[1, 0]), "TP\n" + str(cm[1, 1])],
        ],
        fmt="",
        cmap="Blues",
        xticklabels=["NB(0)", "B(1)"],
        yticklabels=["NB(0)", "B(1)"],
    )
    plt.xlabel("Predicted"), plt.ylabel("Actual"), plt.title(
        "BOLD Cell Prediction CM"
    ), plt.show()

    # Display filtered grids separately
    pred_np, act_np = pred_labels.cpu().numpy(), act_labels.cpu().numpy()

    # Filter predicted DataFrame based on its own predictions
    pred_rows = (pred_np == 1).any(axis=1)
    pred_cols = (pred_np == 1).any(axis=0)
    pred_df_filtered = pd.DataFrame(pred_np).loc[pred_rows, pred_cols]

    # Filter actual DataFrame based on actual labels
    act_rows = (act_np == 1).any(axis=1)
    act_cols = (act_np == 1).any(axis=0)
    act_df_filtered = pd.DataFrame(act_np).loc[act_rows, act_cols]

    print("\n--- Predicted Grid (1 = Bold, 0 = Not Bold) ---")
    if not pred_df_filtered.empty:
        display(pred_df_filtered)
    else:
        print("No bold cells predicted.")

    print("\n--- Actual Grid (1 = Bold, 0 = Not Bold) ---")
    if not act_df_filtered.empty:
        display(act_df_filtered)
    else:
        print("No bold cells in actual data.")


def infer_full(
    trained_model, infer_loader, batch_size=8, threshold=0.5, device="cuda:0"
):
    """
    Runs inference on a dataset with batching, computes average metrics across all files, and handles class imbalance.

    Args:
        trained_model: The trained PyTorch model.
        infer_loader: A dataset-like object containing x_tok and y_tok.
        batch_size: Number of examples per batch for inference.
        threshold: Threshold for classification decision.
        device: Device to run inference on (e.g., "cuda:0" or "cpu").

    Returns:
        None
    """
    # Create DataLoader for batching directly from infer_loader
    batch_loader = torch.utils.data.DataLoader(
        infer_loader, batch_size=batch_size, shuffle=False
    )

    # Initialize cumulative variables
    total_pred_bold = 0
    total_pred_non_bold = 0
    total_act_bold = 0
    total_act_non_bold = 0
    total_confusion_matrix = np.zeros((2, 2), dtype=int)
    cumulative_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    num_batches = len(batch_loader)

    # Set model to eval mode
    trained_model.eval()

    # Iterate through the DataLoader batch-wise
    for batch_idx, batch in enumerate(tqdm(batch_loader, desc="Batch Inference")):

        # Perform inference
        with torch.no_grad():
            pred_probs = torch.sigmoid(trained_model(batch["x_tok"].to(device)))

        # Flatten both labels
        pred_labels = (pred_probs > threshold).long().flatten()

        # Compute metrics for the current batch
        metrics = get_metrics(
            pred_labels, batch["y_tok"][:, :, :, 6].flatten().to(device)
        )

        # Accumulate batch metrics
        total_pred_bold += metrics["pred_bold_count"]
        total_pred_non_bold += metrics["pred_non_bold_count"]
        total_act_bold += metrics["act_bold_count"]
        total_act_non_bold += metrics["act_non_bold_count"]
        total_confusion_matrix += metrics["confusion_matrix"]
        for key in ["accuracy", "precision", "recall", "f1"]:
            cumulative_metrics[key] += metrics[key]

    # Compute average metrics across all batches
    avg_metrics = {
        key: cumulative_metrics[key] / num_batches for key in cumulative_metrics
    }

    # Display aggregated results
    print(f"\n--- Aggregated Metrics Across All Batches ---")
    print(
        f"\nNB to B ratio: Predicted = {total_pred_non_bold}:{total_pred_bold} | "
        f"Actual = {total_act_non_bold}:{total_act_bold}\n"
        f"Accuracy: {avg_metrics['accuracy'] * 100:.2f}% | Precision: {avg_metrics['precision'] * 100:.2f}% | "
        f"Recall: {avg_metrics['recall'] * 100:.2f}% | F1-Score: {avg_metrics['f1']:.2f}\n"
    )

    # Confusion matrix visualization
    cm = total_confusion_matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=[
            ["TN\n" + str(cm[0, 0]), "FP\n" + str(cm[0, 1])],
            ["FN\n" + str(cm[1, 0]), "TP\n" + str(cm[1, 1])],
        ],
        fmt="",
        cmap="Blues",
        xticklabels=["NB(0)", "B(1)"],
        yticklabels=["NB(0)", "B(1)"],
    )
    plt.xlabel("Predicted"), plt.ylabel("Actual"), plt.title(
        "BOLD Cell Prediction CM"
    ), plt.show()


def binfer_one(
    trained_model,
    infer_loader,
    loc=0,
    threshold=0.5,
    condition=">",
    disp_max=False,
    device="cuda:0",
):
    """
    Same functionality as infer_one but adapted for BERT models that require attention masks.
    Displays metrics and visualizations for single example inference without returning values.
    """
    # Set pandas display options for output
    [
        pd.set_option(opt, None) if disp_max else pd.reset_option(opt)
        for opt in ["display.max_rows", "display.max_columns"]
    ]

    # Define valid conditions for prediction thresholding
    conditions = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    if condition not in conditions:
        raise ValueError(
            f"Invalid condition '{condition}'. Must be one of {list(conditions.keys())}."
        )

    # Perform inference with BERT model in evaluation mode
    trained_model.eval()
    with torch.no_grad():
        # Get predictions using both input ids and attention masks
        predictions = trained_model(
            infer_loader.x_tok[loc].unsqueeze(0).to(device),
            infer_loader.x_masks[loc].unsqueeze(0).to(device),
        )

    # Process predictions and get actual labels
    pred_probs = torch.sigmoid(predictions.squeeze(0))
    pred_labels = conditions[condition](pred_probs, threshold).long()
    act_labels = torch.tensor(infer_loader.y_tok[loc][:, :, 6].numpy()).to(device)

    # Calculate metrics for this prediction
    metrics = get_metrics(pred_labels, act_labels)

    # Display filename and prediction details
    print(f"\nFilename: {infer_loader.file_paths[loc]}")

    # Find and display sigmoid values for bold cells
    bold_indices = torch.nonzero(act_labels == 1, as_tuple=False)
    print("\n--- Unique Sigmoid Probabilities for Bold Cells ---")
    if len(bold_indices) > 0:
        unique_sigmoids = {}
        for idx in bold_indices:
            row, col = idx.tolist()
            sigmoid_value = pred_probs[row, col].item()
            if sigmoid_value not in unique_sigmoids:
                unique_sigmoids[sigmoid_value] = (row, col)

        for value, location in sorted(
            unique_sigmoids.items(), key=lambda x: x[0], reverse=False
        ):
            row, col = location
            print(f"({row},{col}): {value:.20f}")
    else:
        print("No bold cells in the actual data.")

    # Display metrics summary
    print(
        f"\nNB to B ratio: Predicted = {metrics['pred_non_bold_count']}:{metrics['pred_bold_count']} | "
        f"Actual = {metrics['act_non_bold_count']}:{metrics['act_bold_count']}\n"
        f"Accuracy: {metrics['accuracy'] * 100:.2f}% | Precision: {metrics['precision'] * 100:.2f}% | "
        f"Recall: {metrics['recall'] * 100:.2f}% | F1-Score: {metrics['f1']:.2f}\n"
    )

    # Create and display confusion matrix visualization
    cm = metrics["confusion_matrix"]
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=[
            ["TN\n" + str(cm[0, 0]), "FP\n" + str(cm[0, 1])],
            ["FN\n" + str(cm[1, 0]), "TP\n" + str(cm[1, 1])],
        ],
        fmt="",
        cmap="Blues",
        xticklabels=["NB(0)", "B(1)"],
        yticklabels=["NB(0)", "B(1)"],
    )
    plt.xlabel("Predicted"), plt.ylabel("Actual"), plt.title(
        "BERT BOLD Cell Prediction CM"
    ), plt.show()

    # Process and display filtered prediction grids
    pred_np, act_np = pred_labels.cpu().numpy(), act_labels.cpu().numpy()

    # Filter predicted grid
    pred_rows = (pred_np == 1).any(axis=1)
    pred_cols = (pred_np == 1).any(axis=0)
    pred_df_filtered = pd.DataFrame(pred_np).loc[pred_rows, pred_cols]

    # Filter actual grid
    act_rows = (act_np == 1).any(axis=1)
    act_cols = (act_np == 1).any(axis=0)
    act_df_filtered = pd.DataFrame(act_np).loc[act_rows, act_cols]

    # Display filtered grids
    print("\n--- Predicted Grid (1 = Bold, 0 = Not Bold) ---")
    if not pred_df_filtered.empty:
        display(pred_df_filtered)
    else:
        print("No bold cells predicted.")

    print("\n--- Actual Grid (1 = Bold, 0 = Not Bold) ---")
    if not act_df_filtered.empty:
        display(act_df_filtered)
    else:
        print("No bold cells in actual data.")


def binfer_full(
    trained_model, infer_loader, batch_size=8, threshold=0.5, device="cuda:0"
):
    """
    Runs inference on a dataset with batching for BERT models, computes average metrics across all files, and handles class imbalance.

    Args:
        trained_model: The trained PyTorch BERT model.
        infer_loader: A dataset-like object containing x_tok, x_masks, and y_tok.
        batch_size: Number of examples per batch for inference.
        threshold: Threshold for classification decision.
        device: Device to run inference on (e.g., "cuda:0" or "cpu").

    Returns:
        None
    """
    # Create DataLoader for batching directly from infer_loader
    batch_loader = torch.utils.data.DataLoader(
        infer_loader, batch_size=batch_size, shuffle=False
    )

    # Initialize cumulative variables
    total_pred_bold = 0
    total_pred_non_bold = 0
    total_act_bold = 0
    total_act_non_bold = 0
    total_confusion_matrix = np.zeros((2, 2), dtype=int)
    cumulative_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    num_batches = len(batch_loader)

    # Set model to eval mode
    trained_model.eval()

    # Iterate through the DataLoader batch-wise
    for batch_idx, batch in enumerate(tqdm(batch_loader, desc="Batch Inference")):

        # Perform inference
        with torch.no_grad():
            # Pass both input ids and attention masks to the model
            pred_probs = torch.sigmoid(
                trained_model(batch["x_tok"].to(device), batch["x_masks"].to(device))
            )

        # Flatten both labels
        pred_labels = (pred_probs > threshold).long().flatten()

        # Compute metrics for the current batch
        metrics = get_metrics(
            pred_labels, batch["y_tok"][:, :, :, 6].flatten().to(device)
        )

        # Accumulate batch metrics
        total_pred_bold += metrics["pred_bold_count"]
        total_pred_non_bold += metrics["pred_non_bold_count"]
        total_act_bold += metrics["act_bold_count"]
        total_act_non_bold += metrics["act_non_bold_count"]
        total_confusion_matrix += metrics["confusion_matrix"]
        for key in ["accuracy", "precision", "recall", "f1"]:
            cumulative_metrics[key] += metrics[key]

    # Compute average metrics across all batches
    avg_metrics = {
        key: cumulative_metrics[key] / num_batches for key in cumulative_metrics
    }

    # Display aggregated results
    print(f"\n--- Aggregated Metrics Across All Batches ---")
    print(
        f"\nNB to B ratio: Predicted = {total_pred_non_bold}:{total_pred_bold} | "
        f"Actual = {total_act_non_bold}:{total_act_bold}\n"
        f"Accuracy: {avg_metrics['accuracy'] * 100:.2f}% | Precision: {avg_metrics['precision'] * 100:.2f}% | "
        f"Recall: {avg_metrics['recall'] * 100:.2f}% | F1-Score: {avg_metrics['f1']:.2f}\n"
    )

    # Confusion matrix visualization
    cm = total_confusion_matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=[
            ["TN\n" + str(cm[0, 0]), "FP\n" + str(cm[0, 1])],
            ["FN\n" + str(cm[1, 0]), "TP\n" + str(cm[1, 1])],
        ],
        fmt="",
        cmap="Blues",
        xticklabels=["NB(0)", "B(1)"],
        yticklabels=["NB(0)", "B(1)"],
    )
    plt.xlabel("Predicted"), plt.ylabel("Actual"), plt.title(
        "BERT BOLD Cell Prediction CM"
    ), plt.show()
