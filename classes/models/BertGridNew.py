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

        # Disable efficient sdp causes problem
        torch.backends.cuda.enable_mem_efficient_sdp(False)

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


"""Expanded Positional Encoding Method"""
# # Newer position encoding method
# def test_posEncoding(self, num_rows, num_cols):
#     """Generates positional encodings for grid cells using sinusoidal functions for each unq 2D combination of row/col."""

#     # Get total number of cells
#     num_cells = num_rows * num_cols

#     # Positional encoding matrix [cells, hidden_size]
#     posEncoding = torch.zeros(num_cells, self.hidden_size, device=self.device)

#     # Create hidden_size vector from 0 to hidden_size - 1 for denom
#     hidden_vec = torch.arange(self.hidden_size, device=self.device)

#     # Set PyTorch print options for better precision display
#     torch.set_printoptions(precision=10, sci_mode=False)

#     # Define columns of interest for debugging
#     debug_rows = [0, 1, 50, 63, 98, 99]  # 1, 63
#     debug_cols = [0, 1, 72, 81, 98, 99]  # 1, 81

#     # Loop through all the number of cell
#     for pos in range(num_cells):

#         # Convert 1D position to 2D coordinates
#         i = pos // num_cols
#         j = pos % num_cols

#         # Only debug print for specific row-column combinations
#         debug_print = (i in debug_rows) and (j in debug_cols)

#         # If col at j is even
#         if j % 2 == 0:

#             # Denom for row and col
#             denoms_row = 10 ** ((4 * 2 * hidden_vec) / (num_rows))
#             denoms_col = 10 ** ((4 * 2 * hidden_vec) / (num_cols))

#             # Apply formula and assign to position
#             pos_vec = torch.sin(i / denoms_row) + torch.sin(j / denoms_col)
#             posEncoding[pos, :] = pos_vec

#             # Debug print for specific even columns (0 and 72)
#             if debug_print:
#                 print(f"\n=== DEBUG INFORMATION FOR EVEN COLUMN {j} ===")
#                 print(f"\nHidden Vec {hidden_vec.shape}:\n{hidden_vec}")
#                 print(f"\n1D: {pos} -> 2D: ({i},{j})")
#                 print(f"\nRow Denoms {denoms_row.shape}:\n{denoms_row}")
#                 print(f"\nCol Denoms {denoms_col.shape}:\n{denoms_col}")
#                 print(f"\nPositional Encoding Vector {pos_vec.shape}:\n{pos_vec}")
#                 print("\n=== END DEBUG INFORMATION ===\n")

#         # Else if col at j is odd
#         else:
#             # 1D denom vector scaled and shifted by 1 from 1 to hidden_size
#             denoms_row = 10 ** ((4 * 2 * (hidden_vec + 1)) / num_rows)
#             denoms_col = 10 ** ((4 * 2 * (hidden_vec + 1)) / num_cols)

#             # Apply formula and assign to position
#             pos_vec = torch.sin(i / denoms_row) + torch.sin(j / denoms_col)
#             posEncoding[pos, :] = pos_vec

#             # Debug print for specific odd columns (1 and 81)
#             if debug_print:
#                 print(f"\n=== DEBUG INFORMATION FOR ODD COLUMN {j} ===")
#                 print(f"\nHidden Vec {hidden_vec.shape}:\n{hidden_vec}")
#                 print(f"\n1D: {pos} -> 2D: ({i},{j})")
#                 print(f"\nRow Denoms {denoms_row.shape}:\n{denoms_row}")
#                 print(f"\nCol Denoms {denoms_col.shape}:\n{denoms_col}")
#                 print(f"\nPositional Encoding Vector {pos_vec.shape}:\n{pos_vec}")
#                 print("\n=== END DEBUG INFORMATION ===\n")

#     # Reset print options to default after we're done
#     torch.set_printoptions(precision=4, sci_mode=False)

#     return posEncoding


"""Expanded New Vectorized Forward Method"""
# # Testing more optimized forward function
#     def forward2(self, input_ids, attention_mask):

#         # Start timing
#         start_time = time.time()

#         # Retrieve dims
#         batch_size, rows, cols, seq_len = input_ids.shape

#         # Initialize [batch, cells, hidden] tensor for embeddings enriched with positional context
#         posContext_embeddings = torch.zeros(
#             (batch_size, rows * cols, self.hidden_size),
#             device=input_ids.device,
#         )

#         # [batch_size * rows * cols, seq_len] = [5 x 100 x 100, 32] = [50000, 32]
#         # Assembles all cells from all sheets in batch and locations in grid in single file
#         # This is done to get BERT embeddings for all cells in one go
#         flattened_input_ids = input_ids.reshape(-1, seq_len)
#         flattened_attention_mask = attention_mask.reshape(-1, seq_len)

#         # Content processed in single shot [batch_size * rows * cols, hidden_size]
#         bert_outputs_flat = self.bertModel_cell(
#             input_ids=flattened_input_ids,
#             attention_mask=flattened_attention_mask,
#         ).pooler_output

#         # Reshape to [batch_size, rows * cols, hidden_size]
#         bert_outputs = bert_outputs_flat.reshape(
#             batch_size, rows * cols, self.hidden_size
#         )

#         # Add [cells, hidden] posEncs to [batch, cells, hidden] BERT embeddings
#         # Broadcasting automatically applies along batch dimension
#         posContext_embeddings = bert_outputs + self.pos_encodings.unsqueeze(0)

#         ## DEBUG PRINT
#         print(f"\nInput IDs: {input_ids.shape}")
#         print(f"Flattened Input IDs: {flattened_input_ids.shape}")
#         print(f"Flattened BERT Pooler Output: {bert_outputs.shape}\n")
#         print(f"PosEncoded Enriched Encodings: {posContext_embeddings.shape}\n")

#         # Process through encoder and classification head, reshape to grid format
#         S_cube = (
#             self.binary_classifier(self.bertEncoder_spatial(posContext_embeddings)[0])
#             .squeeze(-1)
#             .reshape(batch_size, rows, cols)
#         )

#         # Print total time taken
#         print(f"\nNew Forward: {time.time() - start_time:.3f} seconds")

#         # Return the S_cube
#         return S_cube


"""Original Forward Method"""
# # Normal optimized forward function
#     def forward(self, input_ids, attention_mask):
#         """Process input through the model to generate grid cell representations.

#         Takes tokenized input for each cell in the grid, processes it through BERT,
#         combines it with positional information, and generates final cell representations
#         through an encoder and classification head.

#         Args:
#             input_ids (torch.Tensor): Token IDs for each cell in the grid.
#                 Shape: [batch_size, rows, cols, seq_len]
#             attention_mask (torch.Tensor): Attention masks for each cell.
#                 Shape: [batch_size, rows, cols, seq_len]

#         Returns:
#             torch.Tensor: Final representations for each cell in the grid (S_cube).
#                 Shape: [batch_size, rows, cols]
#         """
#         # Retrieve dims and initialize init tensor for posContext embeddings
#         batch_size, rows, cols, seq_len = input_ids.shape
#         posContext_embeddings = torch.zeros(
#             (batch_size, rows * cols, self.hidden_size),
#             device=input_ids.device,
#         )

#         # Build enriched encodings combining content and position understanding
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
#                 + self.pos_encodings[cell]
#             )

#         # Process through encoder and classification head, reshape to grid format
#         S_cube = (
#             self.binary_classifier(self.bertEncoder_spatial(posContext_embeddings)[0])
#             .squeeze(-1)
#             .reshape(batch_size, rows, cols)
#         )

#         # Return the S_cube
#         return S_cube
