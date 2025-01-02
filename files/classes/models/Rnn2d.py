# Imports
import torch
import torch.nn as nn

# Define the base class for RNN-based models with RELU activation
class Rnn2d(nn.Module):
    """
    Base class for RNN-based models operating on 2D grids.

    Subclasses must implement the `cell_hs` method for specific context-handling logic.
    """

    # Initialize shared layers for RNN-based models
    def __init__(self, hidden_state_dim, rnn_layers, embedding_matrix, dropout_rate=0.05, nonlinearity='relu'):
        """
        Initialize shared layers for RNN-based models.

        Args:
            hidden_state_dim (int): Dimension of the hidden state vector.
            rnn_layers (int): Number of RNN layers.
            embedding_matrix (torch.Tensor): Pre-trained embedding matrix.
            dropout_rate (float): Dropout probability for regularization. Defaults to 0.05.
            nonlinearity (str): Type of RNN nonlinearity ('relu' or 'tanh'). Defaults to 'relu'.
        """
        # Call the parent class constructor
        super(Rnn2d, self).__init__()

        # Rows of the embedding matrix = Each word in the vocabulary
        self.vocab_size = embedding_matrix.shape[0]

        # Columns of the embedding matrix = Length of each embedding vector
        self.embedding_dim = embedding_matrix.shape[1]

        # The dimension of the hidden state vector for each step/token
        self.hidden_dim = hidden_state_dim

        # Number of recurrent layers to use
        self.rnn_layers = rnn_layers

        # Create an embedding layer from the pre-trained embedding matrix
        self._embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # Add a dropout layer for regularization
        self._drop = nn.Dropout(dropout_rate)

        # Add an RNN layer with the specified nonlinearity
        self._rnn = nn.RNN(
            self.embedding_dim,
            self.hidden_dim,
            self.rnn_layers,
            nonlinearity=nonlinearity,
            dropout=dropout_rate,
            batch_first=True
        )

        # Add a linear layer to map concatenated hidden states to logits
        self._pred = nn.Linear(2 * self.hidden_dim, 1)

    # Placeholder for cell-specific hidden state calculation
    def cell_hs(self, x):
        """
        Placeholder for cell-specific hidden state calculation.
        Subclasses must implement this method.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, rows, cols, tokens).

        Returns:
            torch.Tensor: Global context tensor.
        """
        # Raise an error if the method is not implemented in the subclass
        raise NotImplementedError("Subclasses must implement the `cell_hs` method.")

    # Define the forward pass for 2D RNN models
    def forward(self, x):
        """
        Forward pass to compute predictions for each cell in the 2D grid.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, rows, cols, tokens).

        Returns:
            torch.Tensor: Predictions for each cell in the 2D grid.
        """
        # Compute global hidden states based on cell-specific logic
        H_global = self.cell_hs(x)  # batch_size x cells x hidden_dim

        # Initialize the output tensor for the 2D grid predictions
        S_cube = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), device=x.device)

        # Loop through all cells in the 2D grid
        for cell in range(x.shape[1] * x.shape[2]):
            # Concatenate local and global contexts, apply dropout, and make predictions
            S_cube[:, cell // x.shape[2], cell % x.shape[2]] = self._pred(
                self._drop(
                    torch.cat(
                        (
                            self._rnn(
                                self._drop(
                                    self._embed(x[:, cell // x.shape[2], cell % x.shape[2], :])
                                )
                            )[0][:, -1, :],
                            H_global[:, cell, :]
                        ),
                        dim=1
                    )
                )
            ).view(-1)

        # Return the final predictions for the 2D grid
        return S_cube

    
    
# Define a subclass that implements context for cells around current in square
class Rnn2dSquare(Rnn2d):
    """
    Subclass for RNN-based models using square context (local hidden state adjustment).
    """

    # Define square-based hidden state logic
    def cell_hs(self, x):
        """
        Compute hidden states for square-based global context.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, rows, cols, tokens).

        Returns:
            torch.Tensor: Global context tensor.
        """
        # Initialize H_local to store hidden states for all cells
        H_local = torch.zeros(x.shape[0], x.shape[1] * x.shape[2], self.hidden_dim, device=x.device)

        # Compute hidden states for each cell and adjust context
        for cell in range(x.shape[1] * x.shape[2]):
            H_local[:, cell, :] = self._rnn(
                self._drop(self._embed(x[:, cell // x.shape[2], cell % x.shape[2], :]))
            )[1][-1]

        # Compute the sum of hidden states and subtract local hidden states
        return H_local.sum(dim=1, keepdim=True) - H_local

    
    
# Define a subclass that implements context for cells around current in a cross
class Rnn2dCross(Rnn2d):
    """
    Subclass for RNN-based models using cross-based global context (row/column adjustment).
    """

    # Define cross-based hidden state logic
    def cell_hs(self, x):
        """
        Compute hidden states for cross-based global context.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, rows, cols, tokens).

        Returns:
            torch.Tensor: Global context tensor.
        """
        # Initialize H_global to store global context for all cells
        H_global = torch.zeros(x.shape[0], x.shape[1] * x.shape[2], self.hidden_dim, device=x.device)

        # Retrieve the grid dimensions
        rows, cols = x.shape[1], x.shape[2]

        # Iterate through all cells to update global context
        for cell in range(rows * cols):
            # Get row indices excluding the current cell
            row_indices = [(cell // cols) * cols + c for c in range(cols) if c != (cell % cols)]

            # Get column indices excluding the current cell
            col_indices = [r * cols + (cell % cols) for r in range(rows) if r != (cell // cols)]

            # Update the global context for the current cell
            H_global[:, row_indices + col_indices, :] += self._rnn(
                self._drop(self._embed(x[:, cell // cols, cell % cols, :]))
            )[1][-1].unsqueeze(1)

        # Return the global context tensor
        return H_global
