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
