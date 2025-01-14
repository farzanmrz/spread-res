# Imports
import os
import time
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import sys
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import copy


def train_model(
    model,
    train_data,
    val_data,
    DEVICE,
    batch_size=8,
    lr=1e-3,
    mu=0.25,
    max_epochs=4,
    patience=2,
    save_int=0,
    save_dir="../models/",
    save_name="model_",
    config=None,
):
    """
    Unified training function that handles both BERT and non-BERT models.
    """

    # Setup logging and paths
    model_path, log_file = setup_logging(save_int, save_dir, save_name, config)

    # Setup training parameters
    (
        opt,
        train_loader,
        val_loader,
        loss_fn,
        epoch,
        best_avgtrloss,
        best_perp,
        best_epoch,
        best_avgvalloss,
        best_valperp,
        nimp_ctr,
        training,
    ) = setup_trainingparams(model, train_data, val_data, lr, batch_size, DEVICE)

    # Main training loop
    while training and (epoch < max_epochs):
        print(f"Epoch {epoch}")
        if save_int > 0:
            with open(log_file, "a") as log:
                log.write(f"\nEpoch {epoch}\n")

        curr_trloss, curr_valloss = 0, 0
        model.train()

        # Training step
        for i, batch in enumerate(tqdm(train_loader, desc="Batch Processing")):

            # Zero the model gradients and set_to_none true to avoid 0 matrices when no grads
            model.zero_grad(set_to_none=True)

            # Get logits/labels as per model type
            logits, labels = get_logitlabels(model, batch, config, DEVICE)

            # Get the loss
            loss = loss_fn(logits, labels)

            # Accumulate to the current training loss
            curr_trloss += loss.detach().cpu().item()

            # Backpropogate loss
            loss.backward()

            # Clip gradients and step optimizer then delete loss param
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=mu)
            opt.step()
            del loss

        # Validation step
        model.eval()
        for i, batch in enumerate(tqdm(val_loader, desc="Validation Processing")):

            # Disable gradients
            with torch.no_grad():

                # Get logits/labels as per model type
                val_logits, val_labels = get_logitlabels(model, batch, config, DEVICE)
                val_loss = loss_fn(val_logits, val_labels)
                curr_valloss += val_loss.detach().cpu().item()

        # Calculate metrics and handle early stopping
        curr_avgtrloss, curr_perp = calculate_metrics(
            curr_trloss, len(train_loader), batch_size
        )
        curr_avgvalloss, curr_valperp = calculate_metrics(
            curr_valloss, len(val_loader), batch_size
        )

        # Get updated best metrics
        (
            best_perp,
            best_valperp,
            best_avgtrloss,
            best_avgvalloss,
            best_epoch,
            nimp_ctr,
        ) = log_metrics(
            curr_avgtrloss,
            curr_perp,
            curr_avgvalloss,
            curr_valperp,
            epoch,
            best_epoch,
            best_avgtrloss,
            best_perp,
            best_avgvalloss,
            best_valperp,
            nimp_ctr,
            save_int,
            log_file,
        )

        # Check if no improvement counter exceeds patience level
        if nimp_ctr >= patience:
            log_metrics(
                curr_avgtrloss,
                curr_perp,
                curr_avgvalloss,
                curr_valperp,
                epoch,
                best_epoch,
                best_avgtrloss,
                best_perp,
                best_avgvalloss,
                best_valperp,
                nimp_ctr,
                save_int,
                log_file,
                early_stopped=True,
            )
            training = False

        # Save model if needed
        if save_int > 0 and (epoch + 1) % save_int == 0:
            torch.save(model.state_dict(), model_path)
            print("Model Saved")
            with open(log_file, "a") as log:
                log.write("Model Saved\n")

        epoch += 1
        print()

    # Final metrics logging
    log_metrics(
        curr_avgtrloss,
        curr_perp,
        curr_avgvalloss,
        curr_valperp,
        epoch - 1,
        best_epoch,
        best_avgtrloss,
        best_perp,
        best_avgvalloss,
        best_valperp,
        nimp_ctr,
        save_int,
        log_file,
        is_final=True,
    )

    return model


def get_logitlabels(model, batch, config, DEVICE):
    """Get logits and labels from a batch based on model type.

    Args:
        model: The model to perform forward pass with
        batch: Current batch of data
        config: Configuration dictionary containing model approach
        DEVICE: Device to run computation on

    Returns:
        tuple: (logits, labels)
    """
    # Forward pass differs based on model type
    if config.get("approach") == "bert":
        logits = model(batch["x_tok"].to(DEVICE), batch["x_masks"].to(DEVICE)).view(-1)
    else:
        logits = model(batch["x_tok"].to(DEVICE)).view(-1)

    labels = batch["y_tok"][:, :, :, 6].to(DEVICE).view(-1).float()
    return logits, labels


def calculate_metrics(loss_sum, num_batches, batch_size):
    """Calculate average loss and perplexity metrics.

    Args:
        loss_sum: Sum of losses across batches
        num_batches: Number of batches
        batch_size: Size of each batch

    Returns:
        tuple: (average_loss, perplexity)
    """
    avg_loss = loss_sum / num_batches
    perplexity = math.exp(loss_sum / (num_batches * batch_size * 2500))
    return avg_loss, perplexity


def setup_logging(save_int, save_dir, save_name, config):
    """Setup logging and model saving paths.

    Args:
        save_int: Saving interval, if 0 then no intermediate saves
        save_dir: Directory to save model and logs
        save_name: Base name for saved files
        config: Configuration dictionary for model

    Returns:
        tuple: (model_path, log_file) or (None, None) if saving disabled
    """
    # Common setup
    torch.set_printoptions(profile="full")

    # Return early if saving disabled
    if save_int == 0:
        return None, None

    # Validate save directory exists
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory '{save_dir}' DNE")

    # Setup paths using timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"{save_name}_{timestamp}.pth")
    log_file = os.path.join(save_dir, f"{save_name}_{timestamp}.txt")

    # Handle config logging if config provided
    if config is not None:

        # Create copy and setup removes list
        config_serializable = copy.deepcopy(config)
        removes = ["DEVICE", "train_loader", "val_loader", "test_loader"]
        removes += (
            ["tokenizer"] if config.get("approach") == "bert" else ["vocab", "wvs"]
        )

        # Remove non-serializable items
        for item in removes:
            config_serializable.pop(item, None)

        # Write config to log file
        with open(log_file, "w") as log:
            log.write("\nFinal configuration:\n")
            log.write(json.dumps(config_serializable, indent=2))
            log.write("\n\n" + "=" * 80 + "\n\n")

    return model_path, log_file


def setup_trainingparams(model, train_data, val_data, lr, batch_size, DEVICE):
    """Setup training parameters including optimizer, dataloaders, loss function and initial tracking variables.

    Args:
        model: The model to train
        train_data: Training dataset
        val_data: Validation dataset
        lr: Learning rate
        batch_size: Batch size for training
        DEVICE: Device to use for training

    Returns:
        tuple: (optimizer, train_loader, val_loader, loss_fn, epoch, best_avgtrloss,
               best_perp, best_epoch, best_avgvalloss, best_valperp, nimp_ctr, training)
    """
    # Setup optimizer and data loaders
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )

    # Setup loss function with class imbalance weighting
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([train_data.get_imbalance()], dtype=torch.float).to(
            DEVICE
        )
    )

    # Initialize training variables
    epoch = 0
    best_avgtrloss = float("inf")
    best_perp = float("inf")
    best_epoch = 0
    best_avgvalloss = float("inf")
    best_valperp = float("inf")
    nimp_ctr = 0
    training = True

    return (
        opt,
        train_loader,
        val_loader,
        loss_fn,
        epoch,
        best_avgtrloss,
        best_perp,
        best_epoch,
        best_avgvalloss,
        best_valperp,
        nimp_ctr,
        training,
    )


def log_metrics(
    curr_avgtrloss,
    curr_perp,
    curr_avgvalloss,
    curr_valperp,
    epoch,
    best_epoch,
    best_avgtrloss,
    best_perp,
    best_avgvalloss,
    best_valperp,
    nimp_ctr,
    save_int,
    log_file,
    is_final=False,
    early_stopped=False,
):
    """Handle metric updates and logging during model training.

    Handles both updating best metric values during training and logging metrics to console
    and file. During regular training updates, returns updated best values. For final or
    early stopping logs, only handles printing/logging without returns.

    Args:
        curr_avgtrloss (float): Current average training loss
        curr_perp (float): Current training perplexity
        curr_avgvalloss (float): Current average validation loss
        curr_valperp (float): Current validation perplexity
        epoch (int): Current epoch number
        best_epoch (int): Best epoch so far
        best_avgtrloss (float): Best average training loss
        best_perp (float): Best training perplexity
        best_avgvalloss (float): Best average validation loss
        best_valperp (float): Best validation perplexity
        nimp_ctr (int): Counter for epochs without improvement
        save_int (int): Save interval for logging (if > 0)
        log_file (str): Path to log file
        is_final (bool, optional): Whether this is final logging. Defaults to False.
        early_stopped (bool, optional): Whether early stopping triggered. Defaults to False.

    Returns:
        tuple or None: If during training updates returns tuple of
        (best_perp, best_valperp, best_avgtrloss, best_avgvalloss, best_epoch, nimp_ctr),
        otherwise None
    """

    # Update best values if improved during regular training
    if not is_final and not early_stopped:
        if curr_valperp < best_valperp:
            # Set new best perplexity values
            best_perp = curr_perp
            best_valperp = curr_valperp

            # Set new best loss values
            best_avgtrloss = curr_avgtrloss
            best_avgvalloss = curr_avgvalloss

            # Update best epoch and reset no improvement counter
            best_epoch = epoch
            nimp_ctr = 0
        else:
            # Increment no improvement counter
            nimp_ctr += 1

    # Set prefix message based on training state
    if early_stopped:
        prefix = f"\nEARLY STOPPING at epoch {epoch}, best epoch {best_epoch}"
    elif is_final:
        prefix = f"\nTRAINING DONE at epoch {epoch}, best epoch {best_epoch}"
    else:
        prefix = f"Epoch {epoch}"

    # Prepare log messages for loss and perplexity
    messages = [
        f"Train Loss = {curr_avgtrloss}, Perplexity = {curr_perp}",
        f"Val Loss = {curr_avgvalloss}, Perplexity = {curr_valperp}\n",
    ]

    # Print prefix and messages to console
    print(prefix)
    for msg in messages:
        print(msg)

    # Write to log file if saving is enabled
    if save_int > 0:
        with open(log_file, "a") as log:
            # Write prefix and messages to log file
            log.write(f"\n{prefix}\n")
            for msg in messages:
                log.write(f"{msg}\n")

    # Return updated values only during training (not for final/early stop logging)
    if not is_final and not early_stopped:
        return (
            best_perp,
            best_valperp,
            best_avgtrloss,
            best_avgvalloss,
            best_epoch,
            nimp_ctr,
        )
