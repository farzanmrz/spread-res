# Imports
import math

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder


# Define the BertPreTiny class
class BertPreTiny(nn.Module):
    """A BERT-based model that combines positional and content understanding for grid-structured data.

    This model processes grid-structured input through BERT embeddings enriched with positional
    encodings for both row and column positions. It uses a combination of BERT encoding and
    positional information to create a rich representation of grid cells.

    Args:
        config (dict): Configuration dictionary containing model parameters.
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

        # Pretrained model for cell's text content and its variables
        self.bertModel_cell = AutoModel.from_pretrained(config["model_base"])

        # Store the hidden size of the pretrained BERT model
        self.pre_hidden_size = self.bertModel_cell.config.hidden_size

        # Project pretrained bert-tiny to our hidden size if different, else use Identity
        self.proj_spatial = (
            nn.Identity()
            if self.bertModel_cell.config.hidden_size == self.hidden_size
            else nn.Linear(self.bertModel_cell.config.hidden_size, self.hidden_size)
        )

        # Custom encoder for row/col position information
        self.bertEncoder_spatial = BertEncoder(self.bert_config)

        # Precompute positional encodings for grid cells
        self.pos_encodings = self.get_posEncoding(self.rows, self.cols)

        # Final binary classification layers
        self.binary_classifier = nn.Sequential(
            nn.Dropout(self.hidden_dropout_prob),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )

    def get_posEncoding(self, num_rows=100, num_cols=100):
        """Generates positional encodings for grid cells using sinusoidal functions.

        Creates a matrix of positional encodings where each row corresponds to a position in the grid
        (either row or column position). Uses sine and cosine functions of different frequencies to
        create unique position embeddings.

        Args:
            num_rows (int, optional): The number of rows in the grid. Defaults to 100.
            num_cols (int, optional): The number of columns in the grid. Defaults to 100.

        Returns:
            torch.Tensor: A tensor of shape [1, num_rows * num_cols, pre_hidden_size] containing
                          the positional encodings, expanded for batch processing.
        """

        # Compute the maximum dimension between rows and columns
        max_dim = max(num_rows, num_cols)

        # Initialize the positional encoding matrix [max_dim, pre_hidden_size]
        posEncoding = torch.zeros(max_dim, self.pre_hidden_size, device=self.device)

        # Create a position vector [max_dim, 1]
        pos = torch.arange(max_dim, dtype=torch.float, device=self.device).unsqueeze(1)

        # Compute frequency scaling factors for sinusoidal encoding
        div_term = torch.exp(
            torch.arange(
                0, self.pre_hidden_size, 2, dtype=torch.float, device=self.device
            )
            * (-math.log(10000.0) / self.pre_hidden_size)
        )

        # Sin = Even, Cos = Odd
        posEncoding[:, 0::2] = torch.sin(pos * div_term)
        posEncoding[:, 1::2] = torch.cos(pos * div_term)

        # Compute a flattened position tensor for row and column indices
        positions = torch.arange(num_rows * num_cols, device=self.device)

        # Return positional encodings expanded for batch processing
        return (
            posEncoding[positions // num_cols] + posEncoding[positions % num_cols]
        ).unsqueeze(0)

    def forward(self, input_ids, attention_mask):
        """Processes input through the model to generate grid cell representations.

        Takes tokenized input for each cell in the grid, processes it through BERT,
        combines it with positional information, and generates final cell representations
        through an encoder and classification head.

        Args:
            input_ids (torch.Tensor): Tensor of shape [batch_size, rows, cols, seq_len]
                                      containing token IDs for each cell in the grid.
            attention_mask (torch.Tensor): Tensor of shape [batch_size, rows, cols, seq_len]
                                           containing attention masks for each cell.

        Returns:
            torch.Tensor: A tensor of shape [batch_size, rows, cols] containing the final
                          binary classification outputs for each cell in the grid.
        """

        # Derive dims
        batch_size, rows, cols, seq_len = input_ids.shape

        # Return S_cube directly calculated
        return (
            self.binary_classifier(
                self.bertEncoder_spatial(
                    self.proj_spatial(
                        self.bertModel_cell(
                            input_ids=input_ids.reshape(
                                batch_size * rows * cols, seq_len
                            ),
                            attention_mask=attention_mask.reshape(
                                batch_size * rows * cols, seq_len
                            ),
                        ).pooler_output.reshape(
                            batch_size, rows * cols, self.pre_hidden_size
                        )
                        + self.pos_encodings.expand(batch_size, -1, -1)
                    )
                )[0]
            )
            .squeeze(-1)
            .reshape(batch_size, rows, cols)
        )


# # Imports
# import math

# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from transformers import AutoModel, BertConfig, BertModel
# from transformers.models.bert.modeling_bert import BertEncoder


# # Define the BertPreTiny class
# class BertPreTiny(nn.Module):
#     """A BERT-based model that combines positional and content understanding for grid-structured data.

#     This model processes grid-structured input through BERT embeddings enriched with positional
#     encodings for both row and column positions. It uses a combination of BERT encoding and
#     positional information to create a rich representation of grid cells.

#     Args:
#     config (dict): Configuration dictionary containing model parameters.
#     """

#     def __init__(self, config):
#         super().__init__()

#         # Extract common params
#         self.device = config["DEVICE"]
#         self.rows = config["rows"]
#         self.cols = config["cols"]
#         self.seq_len = config["tokens"]
#         self.hidden_size = config["hidden_size"]
#         self.hidden_dropout_prob = config["hidden_dropout_prob"]

#         # Create config to be used for both base model and encoder
#         self.bert_config = BertConfig(
#             vocab_size=config["vocab_size"],
#             hidden_size=self.hidden_size,
#             intermediate_size=config["intermediate_size"],
#             num_hidden_layers=config["num_hidden_layers"],
#             num_attention_heads=config["num_attention_heads"],
#             hidden_act=config["hidden_act"],
#             hidden_dropout_prob=self.hidden_dropout_prob,
#             attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
#             max_position_embeddings=config["max_position_embeddings"],
#             type_vocab_size=config["type_vocab_size"],
#             layer_norm_eps=config["layer_norm_eps"],
#             initializer_range=config["initializer_range"],
#             pad_token_id=config["pad_token_id"],
#             gradient_checkpointing=config["gradient_checkpointing"],
#             seed=config["seed"],
#         )

#         # Pretrained model for cell's text content and its variables
#         self.bertModel_cell = AutoModel.from_pretrained(config["model_base"])
#         self.bertModel_cell_hidden_size = self.bertModel_cell.config.hidden_size

#         # Projection layer to put cell output into spatial encoder dims
#         self.proj_spatial = nn.Linear(self.bertModel_cell_hidden_size, self.hidden_size)

#         # Custom encoder for row/col position info
#         self.bertEncoder_spatial = BertEncoder(self.bert_config)

#         # Precompute pos encs for grid cells [max(rows, cols), hidden_size]
#         self.pos_encodings = self.get_posEncoding()

#         # Final binary classification layers wrapped sequentially
#         self.binary_classifier = nn.Sequential(
#             nn.Dropout(self.hidden_dropout_prob),
#             nn.GELU(),
#             nn.Linear(self.hidden_size, 1),
#         )

#     # Function to get positional encodings for cells
#     def get_posEncoding(self):
#         # Max of rows/cols is the number of positions we have
#         max_dim = max(self.rows, self.cols)

#         # Initialize the positional encoding matrix [max_dim, hidden_size]
#         posEncoding = torch.zeros(
#             max_dim, self.bertModel_cell_hidden_size, device=self.device
#         )

#         # Create [max_dim, 1] position vector
#         pos = torch.arange(max_dim, dtype=torch.float, device=self.device).unsqueeze(1)

#         # Compute a [hidden_size/2] vector for the exponential scaling
#         # This replaces repeated pow(10000, 2i/hidden_size) calls
#         div_term = torch.exp(
#             torch.arange(
#                 0,
#                 self.bertModel_cell_hidden_size,
#                 2,
#                 dtype=torch.float,
#                 device=self.device,
#             )
#             * (-math.log(10000.0) / self.bertModel_cell_hidden_size)
#         )

#         # Apply sin to even indices and cos to odd indices
#         posEncoding[:, 0::2] = torch.sin(pos * div_term)
#         posEncoding[:, 1::2] = torch.cos(pos * div_term)

#         # Return final matrix of all positional encodings
#         return posEncoding

#     # Normal optimized forward function
#     def forward(self, input_ids, attention_mask):
#         # Retrieve dims and initialize init tensor for posContext embeddings
#         batch_size, rows, cols, seq_len = input_ids.shape
#         posContext_embeddings = torch.zeros(
#             (batch_size, rows * cols, self.bertModel_cell_hidden_size),
#             device=input_ids.device,
#         )

#         # Build enriched encodings combining content and position understanding
#         # for cell in tqdm(range(rows * cols), desc = 'Doing Forward'):
#         for cell in range(rows * cols):

#             # Define row and column indices for current cell
#             row = cell // self.cols
#             col = cell % self.cols

#             # Calculate the enriched encoding for the cell
#             posContext_embeddings[:, cell, :] = (
#                 self.bertModel_cell(
#                     input_ids=input_ids[:, row, col, :],
#                     attention_mask=attention_mask[:, row, col, :],
#                 ).pooler_output
#                 + self.pos_encodings[row]
#                 + self.pos_encodings[col]
#             )

#         # Process through encoder and classification head, reshape to grid format
#         S_cube = (
#             self.binary_classifier(
#                 self.bertEncoder_spatial(self.proj_spatial(posContext_embeddings))[0]
#             )
#             .squeeze(-1)
#             .reshape(batch_size, rows, cols)
#         )

#         # Return the S_cube
#         return S_cube
