# Imports
import os  # For file and directory operations
import time  # For generating the timestamp in filenames
import torch  # Core PyTorch library
import torch.nn as nn  # For defining loss functions
import math  # For calculating exponential in perplexity calculation
from tqdm import tqdm  # For progress bars in training and validation loops
import sys
from sklearn.metrics import precision_score, recall_score, f1_score


def train_model(model, train_data, val_data, DEVICE, batch_size=8, lr=1.4e-5, mu=0.25, max_epochs=4, patience=3, save_int=2, save_dir='../models/', save_name='rnn_', config=None):
    """
    Train the model for 1 batch, print the length of the train_loader, the training loss, and average training loss.
    """

    # Set the option in torch to print full tensor
    torch.set_printoptions(profile="full")

    # Check if save_int > 0 and save_dir exists
    if save_int > 0 and not os.path.exists(save_dir):
        raise ValueError(f"Directory '{save_dir}' DNE")

    # Define the path where to save model and logfile using save_name as the prefix
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"{save_name}_{timestamp}.pth")
    log_file = os.path.join(save_dir, f"{save_name}_{timestamp}.txt")
    
    # If config exists and save_int > 0, write configuration to log file
    if config is not None and save_int > 0:
        import json
        import copy
        
        # Create a deep copy of the config to avoid modifying the original
        config_serializable = copy.deepcopy(config)
        
        # Remove non-serializable objects
        del config_serializable["DEVICE"] 
        del config_serializable["vocab"]
        del config_serializable["wvs"]
        del config_serializable["train_loader"]
        del config_serializable["val_loader"]
        del config_serializable["test_loader"]
        
        with open(log_file, 'w') as log:
            log.write("\nFinal configuration:\n")
            log.write(json.dumps(config_serializable, indent=2))
            log.write("\n\n" + "="*80 + "\n\n")  

    # Setup optimizer
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)

    # Convert incoming training DataLoader into batches
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Calculate the bold and non-bold cell then the class imbalance (ratio of non-bold to bold cells)
    num_bold_cells = sum((batch['y_tok'][:, :, :, 6] == 1).sum() for batch in train_loader)
    num_nonbold_cells = sum((batch['y_tok'][:, :, :, 6] == 0).sum() for batch in train_loader)
    class_imbalance = num_nonbold_cells / num_bold_cells

    # Spcify the imbalance condition
    imbalance_cond = class_imbalance

    # Binary Cross-Entropy Loss with Logits
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [imbalance_cond],dtype=torch.float
        ).to(DEVICE)
    )

    # Define the starting epoch
    epoch = 0

    # Define the best average training loss, perplexity as inf max value and epoch as 0
    best_avgtrloss = float('inf')
    best_perp = float('inf')
    best_epoch = 0

    # Define the best average val loss, perplexity as inf max value and epoch as 0
    best_avgvalloss = float('inf')
    best_valperp = float('inf')

    # Epochs without improvement counter and flag for training
    nimp_ctr = 0
    training = True

    # Loop while model is in training mode and the epoch is less than max_epochs given
    while training and (epoch < max_epochs):

        # Print the epoch number and write to file also
        print(f'Epoch {epoch}')
        if save_int > 0:
            with open(log_file, 'a') as log:
                log.write(f"\nEpoch {epoch}\n")

        # Initialize training and val losses for current epoch
        curr_trloss,curr_valloss = 0, 0

        ######## TRAIN LOOP ###########

        # Turn on training mode which enables dropout.
        model.train()

        # Loop through the batches in batch_loader
        for i, batch in enumerate(tqdm(train_loader, desc='Batch Processing')):

            # Clear any remaining gradients
            model.zero_grad()

            # Get logits from model prediction, actual labels and the filename required
            logits = model(batch['x_tok'].to(DEVICE)).view(-1)
            labels = batch['y_tok'][:, :, :, 6].to(DEVICE).view(-1).float()

            #Compute loss with mean reduction
            loss = loss_fn(logits, labels)

            # Accumulate the training loss
            curr_trloss += loss.detach().cpu().item()

            # Compute the gradients of the model parameters by backpropagating the loss
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=mu)

            # Update the model parameters
            opt.step()

            # Clear memory
            del loss


        ######## VALIDATION LOOP ###########

        # Turn on eval mode of the model
        model.eval()

        # Loop through the batches in val_loader
        for i, batch in enumerate(tqdm(val_loader, desc='Validation Processing')):

            # Don't track gradients
            with torch.no_grad():

                # Get logits from model prediction, actual labels and the filename required
                logits = model(batch['x_tok'].to(DEVICE)).view(-1)
                labels = batch['y_tok'][:, :, :, 6].to(DEVICE).view(-1).float()

                # Get validation loss
                val_loss = loss_fn(logits, labels)

                # Accumulate the validation loss
                curr_valloss += val_loss.detach().cpu().item()

        # Calculate average training/val loss/perplexity for this epoch
        curr_avgtrloss = curr_trloss / len(train_loader)
        curr_perp = math.exp(curr_trloss / (len(train_loader) * batch_size * 2500))
        curr_avgvalloss = curr_valloss / len(val_loader)
        curr_valperp = math.exp(curr_valloss / (len(val_loader) * batch_size * 2500))

        # Print current stats and log
        print(f'Train Loss: {curr_avgtrloss}, Perplexity: {curr_perp}')
        print(f'Val Loss: {curr_avgvalloss}, Perplexity: {curr_valperp}\n')
        if save_int > 0:
            with open(log_file, 'a') as log:
                log.write(f'Train Loss: {curr_avgtrloss}, Perplexity: {curr_perp}\n')
                log.write(f'Val Loss: {curr_avgvalloss}, Perplexity: {curr_valperp}\n')

        # Update best if current val perp best
        if curr_valperp < best_valperp:
            best_perp = curr_perp
            best_valperp = curr_valperp
            best_avgtrloss = curr_avgtrloss
            best_avgvalloss = curr_avgvalloss
            best_epoch = epoch
            nimp_ctr = 0
        else:
            nimp_ctr+=1


        # Check if patience reached
        if nimp_ctr >= patience:

            # Print early stopping message
            print(f'\nEARLY STOPPING at epoch {epoch}, best epoch {best_epoch}')
            print(f'Train Loss = {best_avgtrloss}, Perplexity = {best_perp}')
            print(f'Val Loss = {best_avgvalloss}, Perplexity = {best_valperp}')

            # Log message
            if save_int > 0:
                with open(log_file, 'a') as log:
                    log.write(f'\nEARLY STOPPING at epoch {epoch}, best epoch {best_epoch}\n')
                    log.write(f'Train Loss = {best_avgtrloss}, Perplexity = {best_perp}\n')
                    log.write(f'Val Loss = {best_avgvalloss}, Perplexity = {best_valperp}\n')

            # Set training to False
            training = False

        # Save the model and log if current epoch is a multiple of save_int
        if save_int > 0 and (epoch + 1) % save_int == 0:
            torch.save(model.state_dict(), model_path)
            print("Model Saved")
            with open(log_file, 'a') as log:
                log.write("Model Saved\n")

        # Increment the epoch and print a new line
        epoch += 1
        print()

    # Save model and log at the end of training (or early stopping)
    if save_int > 0:
        torch.save(model.state_dict(), model_path)

    # Print training complete message
    print(f'\nTRAINING DONE at epoch {epoch-1}, best epoch {best_epoch}')
    print(f'Train Loss = {best_avgtrloss}, Perplexity = {best_perp}')
    print(f'Val Loss = {best_avgvalloss}, Perplexity = {best_valperp}')
    if save_int > 0:
        with open(log_file, 'a') as log:
            log.write(f'\nTRAINING DONE at epoch {epoch-1}, best epoch {best_epoch}\n')
            log.write(f'Train Loss = {best_avgtrloss}, Perplexity = {best_perp}\n')
            log.write(f'Val Loss = {best_avgvalloss}, Perplexity = {best_valperp}\n')

    # Return trained model at the end
    return model


