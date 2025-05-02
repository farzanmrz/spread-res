# Imports
import math  # Keep math import in case needed for future pos encoding

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder


# Define the TestBert base class
class BertGrid(nn.Module):
    """
    A combined BERT-based model for grid-structured data, supporting both
    custom initialization and pre-trained models for cell content encoding.
    """

    def __init__(self, config):
        super().__init__()

        # Extract common params
        self.device = config["DEVICE"]
        self.rows = config["rows"]
        self.cols = config["cols"]
        self.seq_len = config["tokens"]
        self.hidden_size = config["hidden_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]

        # Create config to be used for spatial encoder and potentially custom cell model
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

        # If pretrained tiny then use automodel else custom for config
        if config["model_base"] == "prajjwal1/bert-tiny":
            self.bertModel_cell = AutoModel.from_pretrained(config["model_base"])
        else:
            self.bertModel_cell = BertModel(self.bert_config)

        # Store the hidden size of the cell BERT model
        self.bertModel_cell_hiddenSize = self.bertModel_cell.config.hidden_size

        # Project pretrained/custom bert output to our spatial hidden size if different
        self.proj_spatial = (
            nn.Identity()
            if self.bertModel_cell_hiddenSize == self.hidden_size
            else nn.Linear(self.bertModel_cell_hiddenSize, self.hidden_size)
        )

        # Custom encoder for row/col position information (uses self.bert_config)
        self.bertEncoder_spatial = BertEncoder(self.bert_config)

        # Define the hidden size to use for exponential term
        if config["model_base"] == "prajjwal1/bert-tiny":
            self.exp_hidden_size = self.bertModel_cell_hiddenSize
        else:
            self.exp_hidden_size = self.hidden_size

        # Precompute the exponential term from BertGrid
        self.exp_term = torch.exp(
            -torch.arange(self.exp_hidden_size, device=self.device)
            * torch.log(torch.tensor(10000.0))
            / self.exp_hidden_size
        )

        # Get row/col positional encodings
        self.pos_encodings = self.get_posEncoding(self.rows, self.cols)

        # Final binary classification layers (input dimension is self.hidden_size)
        self.binary_classifier = nn.Sequential(
            nn.Dropout(self.hidden_dropout_prob),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )

    def get_posEncoding(self, num_rows, num_cols):

        # Generate positions tensor
        positions = torch.arange(num_rows * num_cols, device=self.device)

        # Precompute row and column indices just once
        i = (positions // num_cols).unsqueeze(1)
        j = (positions % num_cols).unsqueeze(1)

        # Precompute scaled indices to avoid recomputation
        i_term = i * self.exp_term
        j_term = j * self.exp_term

        # Inline evenness check and sine/cosine selections directly
        return (
            torch.where((i & 1) == 0, torch.sin(i_term), torch.cos(i_term))
            + torch.where((j & 1) == 0, torch.sin(j_term), torch.cos(j_term))
        ).unsqueeze(0)

    def forward(self, input_ids, attention_mask):
        """
        Unified forward pass for the TestBert model.
        """
        # Retrieve dimensions
        batch_size, rows, cols, seq_len = input_ids.shape

        # Generate the S_cube with optimized ordering of operations
        return (
            self.binary_classifier(
                self.bertEncoder_spatial(
                    self.proj_spatial(
                        self.bertModel_cell(
                            input_ids=input_ids.reshape(-1, seq_len),
                            attention_mask=attention_mask.reshape(-1, seq_len),
                        ).pooler_output.reshape(
                            batch_size, rows * cols, self.bertModel_cell_hiddenSize
                        )
                        + self.pos_encodings.to(input_ids.device)
                    )
                )[0]
            )
            .squeeze(-1)
            .reshape(batch_size, rows, cols)
        )
