# Imports
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


def train_model(
    model,
    train_data,
    val_data,
    DEVICE,
    batch_size=8,
    lr=1e-2,
    mu=0.25,
    max_epochs=3,
    patience=2,
    save_int=2,
    save_dir="../models/",
    save_name="model_",
    config=None,
    isPerp=False,
):
    """
    Unified training function that handles both BERT and non-BERT models.
    """

    # ---------- 1. SETUP ----------#

    # 1a. LOGGING MODEL PATH AND FILE
    model_path, log_file = setup_logging(save_int, save_dir, save_name, config)

    # 1b. OPTIMIZER
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)

    # 1c. DATALOADERS
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )

    # 1d. LOSS FUNCTION
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([train_data.get_imbalance()], dtype=torch.float).to(
            DEVICE
        )
    )

    # 1e. INITIAL TRACKING VARIABLES
    epoch = 0
    best_epoch = 0
    nimp_ctr = 0
    best_avgtrloss = float("inf")
    best_avgvalloss = float("inf")
    best_perp = float("inf")
    best_valperp = float("inf")
    isTraining = True

    # ---------- 2. MODEL MAIN LOOP ----------#

    # While isTraining True and max epochs is not reached keep training model
    while isTraining and (epoch < max_epochs):

        # Print epoch number. Log if applicable
        print(f"Epoch {epoch}")
        if save_int > 0:
            with open(log_file, "a") as log:
                log.write(f"\nEpoch {epoch}\n")

        # Define initial values of current train/val loss for the epoch
        curr_trloss = 0
        curr_valloss = 0

        # ---------- 3. TRAINING LOOP ----------#

        # Shift model to training mode
        model.train()

        # Loop through all batches in train loader
        for i, batch in enumerate(tqdm(train_loader, desc="Batch Processing")):

            # Zero the model gradients and avoid 0 matrices when no grads
            model.zero_grad(set_to_none=True)

            # Forward pass through helper function since model type dependent
            logits, labels = get_logitlabels(model, batch, config, DEVICE)

            # Calculate loss using loss function
            loss = loss_fn(logits, labels)

            # Add to current training loss on CPU to reduce GPU memory usage
            curr_trloss += loss.detach().cpu().item()

            # Backpropogate the loss
            loss.backward()

            # Clip gradients using passed mu value
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=mu)

            # Step optimizer
            opt.step()

            # Delete loss parameter to free memory
            del loss

        # ---------- 4. VALIDATION LOOP ----------#

        # Switch model to evaluation mode
        model.eval()

        # Loop through all batches in validation loader
        for i, batch in enumerate(tqdm(val_loader, desc="Validation Processing")):

            # Disable gradients
            with torch.no_grad():

                # Forward pass through helper function since model type dependent
                val_logits, val_labels = get_logitlabels(model, batch, config, DEVICE)

                # Calculate validation loss
                val_loss = loss_fn(val_logits, val_labels)

                # Add to current validation loss on CPU to reduce GPU memory usage
                curr_valloss += val_loss.detach().cpu().item()

        # Calculate metrics and handle early stopping
        curr_avgtrloss = curr_trloss / len(train_loader)
        curr_perp = math.exp(
            curr_trloss
            / (len(train_loader) * batch_size * config["rows"] * config["cols"])
        )
        curr_avgvalloss = curr_valloss / len(val_loader)
        curr_valperp = math.exp(
            curr_valloss
            / (len(val_loader) * batch_size * config["rows"] * config["cols"])
        )

        # Print and log metrics
        if isPerp:
            print(f"Train Loss: {curr_avgtrloss:.4e}, Perplexity: {curr_perp:.4e}")
            print(f"Val Loss: {curr_avgvalloss:.4e}, Perplexity: {curr_valperp:.4e}\n")
        else:
            print(
                f"Train Loss: {curr_avgtrloss:.4e}, Val Loss: {curr_avgvalloss:.4e}\n"
            )
        if save_int > 0:
            with open(log_file, "a") as log:
                log.write(f"Train Loss: {curr_avgtrloss}, Perplexity: {curr_perp}\n")
                log.write(f"Val Loss: {curr_avgvalloss}, Perplexity: {curr_valperp}\n")

        # Early stopping logic
        if curr_valperp < best_valperp:
            best_perp = curr_perp
            best_valperp = curr_valperp
            best_avgtrloss = curr_avgtrloss
            best_avgvalloss = curr_avgvalloss
            best_epoch = epoch
            nimp_ctr = 0
        else:
            nimp_ctr += 1

        if nimp_ctr >= patience:
            print(f"\nEARLY STOPPING at epoch {epoch}, best epoch {best_epoch}")
            if isPerp:
                print(
                    f"Train Loss = {best_avgtrloss:.4e}, Perplexity = {best_perp:.4e}"
                )
                print(
                    f"Val Loss = {best_avgvalloss:.4e}, Perplexity = {best_valperp:.4e}"
                )
            else:
                print(
                    f"Train Loss = {best_avgtrloss:.4e}, Val Loss = {best_avgvalloss:.4e}"
                )
            if save_int > 0:
                with open(log_file, "a") as log:
                    log.write(
                        f"\nEARLY STOPPING at epoch {epoch}, best epoch {best_epoch}\n"
                    )
                    log.write(
                        f"Train Loss = {best_avgtrloss}, Perplexity = {best_perp}\n"
                    )
                    log.write(
                        f"Val Loss = {best_avgvalloss}, Perplexity = {best_valperp}\n"
                    )
            isTraining = False

        # Save model if needed
        if save_int > 0 and (epoch + 1) % save_int == 0:
            torch.save(model.state_dict(), model_path)
            print("Model Saved")
            with open(log_file, "a") as log:
                log.write("Model Saved\n")

        epoch += 1
        print()

    # Final print
    print(f"\nTRAINING DONE at epoch {epoch-1}, best epoch {best_epoch}")
    print(f"Train Loss = {best_avgtrloss}, Perplexity = {best_perp}")
    print(f"Val Loss = {best_avgvalloss}, Perplexity = {best_valperp}")

    # Final save and logging
    if save_int > 0:
        torch.save(model.state_dict(), model_path)
        with open(log_file, "a") as log:
            log.write(f"\nTRAINING DONE at epoch {epoch-1}, best epoch {best_epoch}\n")
            log.write(f"Train Loss = {best_avgtrloss}, Perplexity = {best_perp}\n")
            log.write(f"Val Loss = {best_avgvalloss}, Perplexity = {best_valperp}\n")

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
    # Forward pass differs based on model type for logits
    if config.get("approach") == "bert":
        logits = model(batch["x_tok"].to(DEVICE), batch["x_masks"].to(DEVICE)).view(-1)
    else:
        logits = model(batch["x_tok"].to(DEVICE)).view(-1)

    # Get labels also
    labels = batch["y_tok"][:, :, :, 6].to(DEVICE).view(-1).float()

    # Return both
    return logits, labels


def setup_logging(save_int, save_dir, save_name, config):
    """Setup logging and model saving paths.

    Args:
        save_int: Saving interval, if 0 then no intermediate saves
        save_dir: Directory to save model and logs
        save_name: Base name for saved files
        config: Configuration dictionary for model

    Returns:
        tuple: (model_path, log_file) or (None, None) if saving disable
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

    # Return final modle path and log file
    return model_path, log_file
