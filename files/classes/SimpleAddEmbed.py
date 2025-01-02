import torch
import torch.nn as nn
from tqdm import tqdm


class SimpleAddEmbed(nn.Module):

	def __init__( self, embedding_matrix, dropout_rate = 0.05 ):
		super(SimpleAddEmbed, self).__init__()

		# Rows of embed matrix = Each word in the vocabulary
		self.vocab_size = embedding_matrix.shape[ 0 ]  # vocab_size = 6488

		# Cols of embed matrix = Length of each embedding vector
		self.embedding_dim = embedding_matrix.shape[ 1 ]  # embed_dim = 50

		# Creates an embedding layer from the pre-trained embedding matrix that maps input tokens to their corresponding word vectors
		# If freezing then embeddings don't change during training, we need False because we need them to finetune to our task
		self._embed = nn.Embedding.from_pretrained(embedding_matrix, freeze = False)  # vocab x embed_dim = 6488 x 50

		# Randomly zeroes out a percentage of input units determined by dropout_rate for each update during training
		self._drop = nn.Dropout(dropout_rate)

		# Linear layer to map the concatenated hidden states to logits (1 to predict bold or not)
		self._pred = nn.Linear(self.embedding_dim, 1)

	# Forward function
	def forward( self, x ):

		# Tensor to store the full macro cube of size batch x rows x cols
		S_cube = torch.zeros((x.shape[ 0 ], x.shape[ 1 ], x.shape[ 2 ]), device = x.device)  # 2 x 100 x 100

		# Loop through all rows x cols cells in 1D indexing
		for cell in range(x.shape[ 1 ] * x.shape[ 2 ]):

			# Store into S_cube
			S_cube[ :, cell // x.shape[ 2 ], cell % x.shape[ 2 ] ] = self._pred(

				self._drop(

					self._embed(

						x[ :, cell // x.shape[ 2 ], cell % x.shape[ 2 ], : ]

					).sum(dim = 1)

				)

			).view(-1)

		# Return the final S_cube
		return S_cube
