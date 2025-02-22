# Imports
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder


# Define the BertGridNew class
class BertGridNew(nn.Module):
    """A BERT-based model that combines positional and content understanding for grid-structured data.

    This model processes grid-structured input through BERT embeddings enriched with positional
    encodings for both row and column positions. Using a combination of BERT encoding and positional
    information, it generates a rich representation of grid cells.

    Args:
        config (dict): A configuration dictionary containing model parameters.
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

        # Precompute pos encs for grid cells [max(rows, cols), hidden_size]
        self.pos_encodings = self.get_posEncoding(self.rows, self.cols)

        # Final binary classification layers wrapped sequentially
        self.binary_classifier = nn.Sequential(
            nn.Dropout(self.hidden_dropout_prob),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )

    # Function to get positional encodings for cells
    def get_posEncoding(self, num_rows, num_cols):
        """Generates optimized positional encodings for grid cells.

        Uses sinusoidal functions with broadcasting and combined calculations to efficiently compute positional
        encodings while preserving the mathematical properties of the original implementation.

        Args:
            num_rows (int): The number of rows in the grid.
            num_cols (int): The number of columns in the grid.

        Returns:
            torch.Tensor: A tensor of shape [1, num_rows * num_cols, hidden_size] containing the positional encodings.
        """
        # Calculate all position indices at once to get row/col indices
        positions = torch.arange(num_rows * num_cols, device=self.device)
        i = positions // num_cols
        j = positions % num_cols

        # Create hidden dimension vector for frequency calculations
        frequency_base = 8.0 * (
            torch.arange(self.hidden_size, device=self.device) + (j[:, None] % 2)
        )

        # Return [batch_size, rows * cols, hidden_size] tensor with formula
        return (
            (torch.sin(i[:, None] / (10 ** (frequency_base / num_rows))))
            + (torch.sin(j[:, None] / (10 ** (frequency_base / num_cols))))
        ).unsqueeze(0)

    def forward(self, input_ids, attention_mask):
        """Process input through the model to generate grid cell representations.

        This method reshapes the input tokens for each grid cell, obtains BERT embeddings for the cells,
        and then enriches them with precomputed positional encodings. The enriched representations are
        passed through a spatial encoder and a binary classifier to produce a grid-shaped output.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs with shape [batch_size, rows, cols, seq_len].
            attention_mask (torch.Tensor): Tensor of attention masks with shape [batch_size, rows, cols, seq_len].

        Returns:
            torch.Tensor: Tensor of shape [batch_size, rows, cols] containing the binary classification
                          outputs for each grid cell.
        """
        # Retrieve dims
        batch_size, rows, cols, seq_len = input_ids.shape

        # Generate the S_cube and return directly
        return (
            self.binary_classifier(
                self.bertEncoder_spatial(
                    (
                        (
                            self.bertModel_cell(
                                input_ids=input_ids.reshape(-1, seq_len),
                                attention_mask=attention_mask.reshape(-1, seq_len),
                            ).pooler_output.reshape(
                                batch_size, rows * cols, self.hidden_size
                            )
                        )
                        + (self.pos_encodings.to(input_ids.device))
                    )
                )[0]
            )
            .squeeze(-1)
            .reshape(batch_size, rows, cols)
        )
