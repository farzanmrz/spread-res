# Import commands
import operator
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.cuda import set_device
from tqdm import tqdm

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
    device="cuda:0",
    approach="bert",
):
    # Set pandas display options
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    # Switch model to evaluation mode
    trained_model.eval()

    # Disable gradient computation for inference
    with torch.no_grad():

        # Get predictions based on the model approach
        predictions = get_single_pred(
            trained_model, infer_loader, loc, device, approach
        )

    # Convert predictions to probabilities using sigmoid
    pred_probs = torch.sigmoid(predictions.squeeze(0))

    # Get the predicted labels as determined by the threshold
    pred_labels = (pred_probs > threshold).long()

    # Get the actual labels from the loader
    act_labels = torch.tensor(infer_loader.y_tok[loc][:, :, 6].numpy()).to(device)

    # Calculate metrics to display using predicted and actual labels
    metrics = get_metrics(pred_labels, act_labels)

    # Print filename
    print(f"\nFilename: {infer_loader.file_paths[loc]}")

    ########## BOLD CELLS UNIQUE SIGMOID VALUE DISPLAY ##############

    # Get indices for bold cells in actual data
    bold_indices = torch.nonzero(act_labels == 1, as_tuple=False)

    # Print message for showing unique sigmoid values for bold cells
    print("\n--- Unique Sigmoid Probabilities for Bold Cells ---")

    # If there are bold cells in the actual data
    if len(bold_indices) > 0:

        # Create a dict to store unique sigmoid values with example location
        unique_sigmoids = {}

        # Iterate through bold cell indices
        for idx in bold_indices:

            # Get row and column indices
            row, col = idx.tolist()

            # Get the sigmoid value
            sigmoid_value = pred_probs[row, col].item()

            # If value not in unique_sigmoids
            if sigmoid_value not in unique_sigmoids:

                # Store the sigmoid value with the location
                unique_sigmoids[sigmoid_value] = (row, col)

        # Sort the dictionary by sigmoid values in ascending order
        sorted_sigmoids = sorted(
            unique_sigmoids.items(), key=lambda x: x[0], reverse=False
        )

        # Print unique sigmoid values with example locations
        for value, location in sorted_sigmoids:

            # Get row and column indices
            row, col = location

            # Print the location and sigmoid value
            print(f"({row},{col}): {value:.20f}")

    # Else, if no bold cells in the actual data
    else:

        # Print message
        print("No bold cells in the actual data.")

    # Display metrics summary
    print(
        f"\nNB to B ratio: Predicted = {metrics['pred_non_bold_count']}:{metrics['pred_bold_count']} | "
        f"Actual = {metrics['act_non_bold_count']}:{metrics['act_bold_count']}\n"
        f"Accuracy: {metrics['accuracy'] * 100:.2f}% | Precision: {metrics['precision'] * 100:.2f}% | "
        f"Recall: {metrics['recall'] * 100:.2f}% | F1-Score: {metrics['f1']:.2f}\n"
    )

    # Retrive confusion matrix from metrics
    cm = metrics["confusion_matrix"]

    # Define the plot size for the confusion matrix
    plt.figure(figsize=(6, 4))

    # Create a heatmap for the confusion matrix
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

    # Set the labels and title for the confusion matrix
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("BOLD Cell Prediction CM")

    # Display the confusion matrix
    plt.show()

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
    trained_model,
    infer_loader,
    batch_size=8,
    threshold=0.5,
    device="cuda:0",
    approach="bert",
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
            pred_probs = torch.sigmoid(
                get_multi_pred(trained_model, batch, device, approach)
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
        "BOLD Cell Prediction CM"
    ), plt.show()


def get_single_pred(trained_model, infer_loader, loc, device, approach):
    """
    Get predictions based on the model approach.

    Args:
        trained_model: The trained model for inference.
        infer_loader: Data loader containing data to infer.
        loc: Index of the file to infer on.
        device: Device for computation (e.g., 'cuda:0', 'cpu').
        approach: Model type ('simple' or 'bert').

    Returns:
        torch.Tensor: Predicted logits from the model.
    """
    if approach == "bert":
        # Perform BERT-specific inference using input IDs and attention masks
        predictions = trained_model(
            infer_loader.x_tok[loc].unsqueeze(0).to(device),
            infer_loader.x_masks[loc].unsqueeze(0).to(device),
        )
    elif approach == "simple":
        # Perform simple model inference using only input IDs
        predictions = trained_model(infer_loader.x_tok[loc].unsqueeze(0).to(device))
    else:
        raise ValueError(f"Unsupported approach: {approach}")
    return predictions


def get_multi_pred(trained_model, batch, device, approach):
    """
    Get predictions for a full batch based on the model approach.

    Args:
        trained_model: The trained model for inference.
        batch: Batch of data containing x_tok, x_masks (if BERT), and y_tok.
        device: Device for computation (e.g., 'cuda:0', 'cpu').
        approach: Model type ('simple' or 'bert').

    Returns:
        torch.Tensor: Predicted logits for the batch.
    """
    if approach == "bert":
        # Perform BERT-specific inference using input IDs and attention masks
        predictions = trained_model(
            batch["x_tok"].to(device),
            batch["x_masks"].to(device),
        )
    elif approach == "simple":
        # Perform simple model inference using only input IDs
        predictions = trained_model(batch["x_tok"].to(device))
    else:
        raise ValueError(f"Unsupported approach: {approach}")
    return predictions
