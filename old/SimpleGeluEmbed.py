# Imports
import torch
import torch.nn as nn

# Define the base class for embedding-based models with GELU activation
class SimpleGeluEmbed(nn.Module):
    """
    Base class for simple embedding-based models with GELU activation.

    This class handles shared functionality such as embedding, dropout, and GELU non-linearity.
    Subclasses must implement the `aggregate` method for specific aggregation logic (e.g., sum or average).
    """

    # Initialize shared layers for embedding-based models
    def __init__(self, embedding_matrix, dropout_rate=0.05):
        """
        Initializes the SimpleGeluEmbed class with shared layers.

        Args:
            embedding_matrix (torch.Tensor): Pre-trained embedding matrix of shape (vocab_size, embedding_dim).
            dropout_rate (float): Dropout probability for regularization. Defaults to 0.05.
        """
        
        # Call the parent class constructor
        super(SimpleGeluEmbed, self).__init__()

        # Store the vocabulary size from the embedding matrix
        self.vocab_size = embedding_matrix.shape[0]

        # Store the embedding dimension from the embedding matrix
        self.embedding_dim = embedding_matrix.shape[1]

        # Create an embedding layer using the pre-trained embedding matrix
        self._embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # Create a dropout layer for regularization
        self._drop = nn.Dropout(dropout_rate)

        # Create a GELU non-linearity layer
        self._non_linear = nn.GELU()

        # Create a linear layer to predict logits
        self._pred = nn.Linear(self.embedding_dim, 1)

    # Define the forward pass for processing input tensors
    def forward(self, x):
        """
        Processes input tensor and returns logits for each cell in the grid.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, rows, cols, tokens).

        Returns:
            torch.Tensor: Output logits for each cell, of shape (batch_size, rows, cols).
        """

        # Initialize an empty tensor to store output logits for each cell
        S_cube = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), device=x.device)

        # Iterate over all cells in the 2D grid (flattened into 1D indexing)
        for cell in range(x.shape[1] * x.shape[2]):

            # Compute logits for the current cell and store in S_cube
            S_cube[:, cell // x.shape[2], cell % x.shape[2]] = self._pred(
                self._non_linear(
                    self._drop(
                        self.aggregate(
                            self._embed(x[:, cell // x.shape[2], cell % x.shape[2], :])
                        )
                    )
                )
            ).view(-1)

        # Return the logits tensor for all cells
        return S_cube

    # Placeholder for the aggregation method to be implemented by subclasses
    def aggregate(self, embeddings):
        """
        Placeholder for aggregation logic (e.g., sum or average).

        Args:
            embeddings (torch.Tensor): Tensor of shape (batch_size, tokens, embedding_dim).

        Returns:
            torch.Tensor: Aggregated embeddings.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        
        # Raise an error if this method is not implemented in a subclass
        raise NotImplementedError("Subclasses must define the aggregate method.")

# Define a subclass that implements summing for aggregation
class SimpleGeluEmbedAdd(SimpleGeluEmbed):
    """
    Model subclass for summing embeddings along the token dimension.
    Inherits shared functionality from SimpleGeluEmbed.
    """

    # Implement the aggregate method for summing embeddings
    def aggregate(self, embeddings):
        """
        Aggregates embeddings by summing along the token dimension.

        Args:
            embeddings (torch.Tensor): Tensor of shape (batch_size, tokens, embedding_dim).

        Returns:
            torch.Tensor: Aggregated embeddings of shape (batch_size, embedding_dim).
        """
        
        # Return the sum of embeddings along the token dimension
        return embeddings.sum(dim=1)

# Define a subclass that implements averaging for aggregation
class SimpleGeluEmbedAvg(SimpleGeluEmbed):
    """
    Model subclass for averaging embeddings along the token dimension.
    Inherits shared functionality from SimpleGeluEmbed.
    """

    # Implement the aggregate method for averaging embeddings
    def aggregate(self, embeddings):
        """
        Aggregates embeddings by averaging along the token dimension.

        Args:
            embeddings (torch.Tensor): Tensor of shape (batch_size, tokens, embedding_dim).

        Returns:
            torch.Tensor: Aggregated embeddings of shape (batch_size, embedding_dim).
        """
        
        # Return the average of embeddings along the token dimension
        return embeddings.mean(dim=1)

    
    
############# CHECKER CODE FOR POSITION BASED EMBEDDING ###########
###################################################################

# import torch
# import torch.nn as nn
# from tqdm import tqdm


# class PosGeluAvgEmbed(nn.Module):

#     # Initialize constructor
#     def __init__(self, embedding_matrix, dropout_rate=0.05):

#         super(PosGeluAvgEmbed, self).__init__()

#         # Get the number of words in the vocabulary
#         self.vocab_size = embedding_matrix.shape[0]

#         # Get the embedding dimension for each word
#         self.embedding_dim = embedding_matrix.shape[1]

#         # Define an embedding layer initialized with pre-trained embeddings
#         self._embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

#         # Define a dropout layer to prevent overfitting
#         self._drop = nn.Dropout(dropout_rate)

#         # Define a GELU activation layer for non-linearity
#         self._non_linear = nn.GELU()

#         # Define a linear layer to predict logits for bold/non-bold classification
#         self._pred = nn.Linear(self.embedding_dim, 1)



#     # def pos_embed(self, index):
#     #     """
#     #     Generate sinusoidal positional embeddings for a given row/column index.

#     #     Args:
#     #         index (int): Row or column index for which positional embedding is calculated.

#     #     Returns:
#     #         torch.Tensor: Positional embedding vector of size (embedding_dim,).
#     #     """
#     #     # Get positional dimension k
#     #     k = torch.arange(0, self.embedding_dim, dtype=torch.float32, device=index.device)

#     #     # Get the denominator
#     #     denom = 10000 ** (k / self.embedding_dim)

#     #     # Apply sin and get final position embedding
#     #     pos_embedding = torch.sin(index.float() / denom)

#     #     # Return the final position embedding
#     #     return pos_embedding

#     def pos_embed(self, index, axis, device):
#         """
#         Calculate sinusoidal positional embeddings, alternating sine and cosine for rows and columns.

#         Args:
#             index (int): Row or column index for which positional embedding is calculated.
#             axis (str): 'row' or 'col' to determine which function to apply.
#             device (torch.device): Device on which to calculate the embeddings.

#         Returns:
#             torch.Tensor: Positional embedding vector of size (embedding_dim,).
#         """
#         # Get positional dimension k
#         k = torch.arange(0, self.embedding_dim, dtype=torch.float32, device=device)

#         # Get the denominator
#         denom = 10000 ** (k / self.embedding_dim)

#         # Apply sine for rows and cosine for columns
#         if axis == 'row':
#             pos_embedding = torch.sin(index / denom)  # Use sine for rows
#         elif axis == 'col':
#             pos_embedding = torch.cos(index / denom)  # Use cosine for columns
#         else:
#             raise ValueError("Invalid axis value. Must be 'row' or 'col'.")

#         return pos_embedding



#     def forward(self, x):

#         # Initialize an empty tensor to store predictions for all cells
#         S_cube = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), device=x.device)

#         # Iterate over all cells in the 2D grid (100x100 cells)
#         for cell in range(x.shape[1] * x.shape[2]):

#             # Calculate the row and column indices for the current cell
#             row = cell // x.shape[2]
#             col = cell % x.shape[2]

#             # Calculate the positional encodings for row/col
#             row_pos_embed = self.pos_embed(row, axis='row', device=x.device)  # Sine for rows
#             col_pos_embed = self.pos_embed(col, axis='col', device=x.device)

#             # Extract token embeddings for the current cell
#             token_embeddings = self._embed(x[:, row, col, :])  # Shape: batch x seq_len x embed_dim

#             # Compute the average of token embeddings for the current cell
#             averaged_embedding = token_embeddings.mean(dim=1)  # Shape: batch x embed_dim

#             # Now compute the enriched embedding with the position embeddings added
#             enriched_embedding = averaged_embedding + row_pos_embed + col_pos_embed

#             # Apply dropout for regularization
#             dropped_embedding = self._drop(enriched_embedding)  # Shape: batch x embed_dim

#             # Apply GELU activation for non-linearity
#             activated_embedding = self._non_linear(dropped_embedding)  # Shape: batch x embed_dim

#             # Predict logits for bold/non-bold classification
#             logits = self._pred(activated_embedding)  # Shape: batch x 1

#             # Store the logits in the S_cube tensor
#             S_cube[:, row, col] = logits.view(-1)

#         # Return the final predictions for all cells
#         return S_cube
