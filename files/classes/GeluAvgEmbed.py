import torch
import torch.nn as nn
from tqdm import tqdm


class GeluAvgEmbed(nn.Module):
    def __init__(self, embedding_matrix, dropout_rate=0.05):
        super(GeluAvgEmbed, self).__init__()

        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_dim = embedding_matrix.shape[1]
        self._embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self._drop = nn.Dropout(dropout_rate)
        self._non_linear = nn.GELU()  # Use GELU instead of tanh
        self._pred = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        S_cube = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), device=x.device)
        for cell in range(x.shape[1] * x.shape[2]):
            S_cube[:, cell // x.shape[2], cell % x.shape[2]] = self._pred(
                self._non_linear(
                    self._drop(
                        self._embed(
                            x[:, cell // x.shape[2], cell % x.shape[2], :]
                        ).mean(dim=1)
                    )
                )
            ).view(-1)
        return S_cube
