# Import required libraries
from transformers import BertModel, BertConfig
import torch.nn as nn
import torch


# Define the BertPooler class
class BertPooler(nn.Module):
    """A BERT-based model for processing grid-structured input data.

    This model uses BERT to process text in a grid format, applying the model to each cell
    independently and producing a 3D tensor of scores.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Dimension of hidden layers.
        intermediate_size (int): Size of intermediate layers.
        num_hidden_layers (int): Number of transformer layers.
        num_attention_heads (int): Number of attention heads.
        hidden_act (str): Activation function for hidden layers.
        hidden_dropout_prob (float): Dropout probability for hidden layers.
        attention_probs_dropout_prob (float): Dropout probability for attention.
        max_position_embeddings (int): Maximum sequence length.
        type_vocab_size (int): Size of token type vocabulary.
        layer_norm_eps (float): Layer normalization epsilon.
        initializer_range (float): Range for weight initialization.
        pad_token_id (int): ID of padding token.
        gradient_checkpointing (bool): Whether to use gradient checkpointing.
    """

    def __init__(self, config):
        """Initialize the BertPooler model.

        Args:
            config (dict): Configuration dictionary with model parameters.
        """
        # Initialize parent class
        super().__init__()

        # Extract configuration parameters from config dictionary
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_attention_heads = config["num_attention_heads"]
        self.hidden_act = config["hidden_act"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attention_probs_dropout_prob = config["attention_probs_dropout_prob"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.type_vocab_size = config["type_vocab_size"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.pad_token_id = config["pad_token_id"]
        self.gradient_checkpointing = config["gradient_checkpointing"]

        # Create BERT configuration object with extracted parameters
        self.config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        # Initialize BERT model with the configuration
        self.bert = BertModel(self.config)

        # Setup classification components
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.gelu = nn.GELU()
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Process input through the BERT model and classification head.

        Args:
            input_ids (torch.Tensor): Input token ids of shape (batch_size, rows, cols, seq_len)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, rows, cols, seq_len)

        Returns:
            torch.Tensor: Output scores of shape (batch_size, rows, cols)
        """
        # Extract dimensions from input
        batch_size, rows, cols, seq_len = input_ids.shape

        # Initialize output tensor
        S_cube = torch.zeros((batch_size, rows, cols), device=input_ids.device)

        # Process each cell in the grid
        for cell in range(rows * cols):
            # Calculate row and column indices
            row = cell // cols
            col = cell % cols

            # Combined classification head operations into single inline expression
            S_cube[:, row, col] = self.classifier(
                self.gelu(
                    self.dropout(
                        self.bert(
                            input_ids=input_ids[:, row, col, :],
                            attention_mask=attention_mask[:, row, col, :],
                        ).pooler_output
                    )
                )
            ).squeeze(-1)

        return S_cube
