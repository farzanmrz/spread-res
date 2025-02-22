# Imports
import math

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder


# Define the BertGrid class
class BertGrid(nn.Module):
    """A BERT-based model that combines positional and content understanding for grid-structured data.

    This model processes grid-structured input through BERT embeddings enriched with positional
    encodings for both row and column positions. It uses a combination of BERT encoding and
    positional information to create a rich representation of grid cells.

    Args:
    config (dict): Configuration dictionary containing model parameters.
    """

    def __init__(self, config):
        super().__init__()

        # Disable efficient sdp globally
        # torch.backends.cuda.enable_mem_efficient_sdp(False)

        # Extract common params
        self.device = config["DEVICE"]
        self.rows = config["rows"]
        self.cols = config["cols"]
        self.seq_len = config["tokens"]
        self.hidden_size = config["hidden_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]

        # Create config to be used for both base model and encoder
        self.bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=self.hidden_size,
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            hidden_act=config["hidden_act"],
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
            max_position_embeddings=config["max_position_embeddings"],
            type_vocab_size=config["type_vocab_size"],
            layer_norm_eps=config["layer_norm_eps"],
            initializer_range=config["initializer_range"],
            pad_token_id=config["pad_token_id"],
            gradient_checkpointing=config["gradient_checkpointing"],
            seed=config["seed"],
        )

        # Initialize both the BERT model and the enriched encoder using the same config.
        self.bertModel_cell = BertModel(self.bert_config)
        self.bertEncoder_spatial = BertEncoder(self.bert_config)

        # Precompute both versions of positional encodings
        self.pos_encodings = self.get_posEncoding(self.rows, self.cols)

        # Final binary classification layers wrapped sequentially
        self.binary_classifier = nn.Sequential(
            nn.Dropout(self.hidden_dropout_prob),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )

    # Function to get positional encodings for cells
    def get_posEncoding(self, num_rows=100, num_cols=100):
        """Generates positional encodings for grid cells using sinusoidal functions.

        Args:
            num_rows (int, optional): Number of rows. Defaults to 100
            num_cols (int, optional): Number of columns. Defaults to 100

        Returns:
            torch.Tensor: Tensor of shape [1, rows*cols, hidden_size] containing combined
                        positional encodings for each grid position
        """
        # Initialize the positional encoding matrix [max_dim, hidden_size]
        max_dim = max(num_rows, num_cols)
        posEncoding = torch.zeros(max_dim, self.hidden_size, device=self.device)

        # Create position vector and frequency terms
        pos = torch.arange(max_dim, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, dtype=torch.float, device=self.device)
            * (-math.log(10000.0) / self.hidden_size)
        )

        # Apply sin to even indices and cos to odd indices
        posEncoding[:, 0::2] = torch.sin(pos * div_term)
        posEncoding[:, 1::2] = torch.cos(pos * div_term)

        # Get indices for all grid positions
        positions = torch.arange(num_rows * num_cols, device=self.device)
        row_indices = positions // num_cols
        col_indices = positions % num_cols

        # Return combined row and column encodings [1, rows*cols, hidden_size]
        return (posEncoding[row_indices] + posEncoding[col_indices]).unsqueeze(0)

    # New fully optimized forward function
    def forward(self, input_ids, attention_mask):
        """Processes input through the model with fully vectorized operations."""
        batch_size, rows, cols, seq_len = input_ids.shape

        # Return the S_cube directly using pre-computed pos_encodings
        return (
            self.binary_classifier(
                self.bertEncoder_spatial(
                    (
                        self.bertModel_cell(
                            input_ids=input_ids.reshape(-1, seq_len),
                            attention_mask=attention_mask.reshape(-1, seq_len),
                        ).pooler_output.reshape(
                            batch_size, rows * cols, self.hidden_size
                        )
                    )
                    + self.pos_encodings.expand(batch_size, -1, -1)
                )[0]
            )
            .squeeze(-1)
            .reshape(batch_size, rows, cols)
        )
