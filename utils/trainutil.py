# Imports
import os
import time
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import sys
from sklearn.metrics import precision_score, recall_score, f1_score


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
    # Common setup
    torch.set_printoptions(profile="full")

    if save_int > 0 and not os.path.exists(save_dir):
        raise ValueError(f"Directory '{save_dir}' DNE")

    # Setup paths and logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"{save_name}_{timestamp}.pth")
    log_file = os.path.join(save_dir, f"{save_name}_{timestamp}.txt")

    # Handle config logging based on model type
    if config is not None and save_int > 0:
        import json
        import copy

        config_serializable = copy.deepcopy(config)

        # Remove non-serializable items based on model type
        common_removes = ["DEVICE", "train_loader", "val_loader", "test_loader"]
        bert_removes = ["tokenizer"]
        simple_removes = ["vocab", "wvs"]

        # Remove common items
        for item in common_removes:
            config_serializable.pop(item, None)

        # Remove model-specific items
        if config.get("approach") == "bert":
            for item in bert_removes:
                config_serializable.pop(item, None)
        else:
            for item in simple_removes:
                config_serializable.pop(item, None)

        with open(log_file, "w") as log:
            log.write("\nFinal configuration:\n")
            log.write(json.dumps(config_serializable, indent=2))
            log.write("\n\n" + "=" * 80 + "\n\n")

    # Setup optimizer and data loaders
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )

    # Setup loss function with ratio of nb to b cells retrieved from original loader for training
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
            model.zero_grad()

            # Forward pass differs based on model type
            if config.get("approach") == "bert":
                logits = model(
                    batch["x_tok"].to(DEVICE), batch["x_masks"].to(DEVICE)
                ).view(-1)
            else:
                logits = model(batch["x_tok"].to(DEVICE)).view(-1)

            labels = batch["y_tok"][:, :, :, 6].to(DEVICE).view(-1).float()
            loss = loss_fn(logits, labels)
            curr_trloss += loss.detach().cpu().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=mu)
            opt.step()
            del loss

        # Validation step
        model.eval()
        for i, batch in enumerate(tqdm(val_loader, desc="Validation Processing")):
            with torch.no_grad():
                if config.get("approach") == "bert":
                    val_logits = model(
                        batch["x_tok"].to(DEVICE), batch["x_masks"].to(DEVICE)
                    ).view(-1)
                else:
                    val_logits = model(batch["x_tok"].to(DEVICE)).view(-1)

                val_labels = batch["y_tok"][:, :, :, 6].to(DEVICE).view(-1).float()
                val_loss = loss_fn(val_logits, val_labels)
                curr_valloss += val_loss.detach().cpu().item()

        # Calculate metrics and handle early stopping
        curr_avgtrloss = curr_trloss / len(train_loader)
        curr_perp = math.exp(curr_trloss / (len(train_loader) * batch_size * 2500))
        curr_avgvalloss = curr_valloss / len(val_loader)
        curr_valperp = math.exp(curr_valloss / (len(val_loader) * batch_size * 2500))

        # Print and log metrics
        print(f"Train Loss: {curr_avgtrloss}, Perplexity: {curr_perp}")
        print(f"Val Loss: {curr_avgvalloss}, Perplexity: {curr_valperp}\n")
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
            print(f"Train Loss = {best_avgtrloss}, Perplexity = {best_perp}")
            print(f"Val Loss = {best_avgvalloss}, Perplexity = {best_valperp}")
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
            training = False

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
