import torch
import torch.nn as nn

class TestRNN(nn.Module):

    # Constructor of the RNN_LM class, initializing the layers and weights
    def __init__(self, hidden_state_dim, rnn_layers, embedding_matrix, dropout_rate=0.05, nonlinearity='relu'):

        # Ensures functions of parent class nn.Module are called in subclass RNN_LM
        super(TestRNN, self).__init__()

        # Rows of embed matrix = Each word in the vocabulary
        self.vocab_size = embedding_matrix.shape[0]  # vocab_size = 34057

        # Cols of embed matrix = Length of each embedding vector
        self.embedding_dim = embedding_matrix.shape[1]  # embed_dim = 50

        # The dimension of the hidden state vector 'h' for each step/token
        self.hidden_dim = hidden_state_dim  # hid_dim = 100

        # Number of recurrent layers we will use
        self.rnn_layers = rnn_layers  # rnn_layers = 2

        # Creates an embedding layer from the pre-trained embedding matrix that maps input tokens to their corresponding word vectors
        # If freezing then embeddings don't change during training, we need False because we need them to finetune to our task
        self._embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # Randomly zeroes out a percentage of input units determined by dropout_rate for each update during training
        self._drop = nn.Dropout(dropout_rate)

        # RNN layer with 'relu' nonlinearity but not managing exploding gradients, dropout and multiple recurrent layers
        self._rnn = nn.RNN(
            self.embedding_dim,
            self.hidden_dim,
            self.rnn_layers,
            nonlinearity=nonlinearity,
            dropout=dropout_rate,
            batch_first = True
        )

        # Linear layer to map the concatenated hidden states to logits (1 to predict bold or not)
        self._pred = nn.Linear(2 * self.hidden_dim, 1)

    def cell_hs(self, x):

        # Set the manual seed for reproducibility
        torch.manual_seed(0)

        # Initialize H_local as a zero tensor with the appropriate shape (num_cells, hidden_dim)
        H_local = torch.zeros(x.shape[0], x.shape[1] * x.shape[2], self.hidden_dim, device=x.device) # batch x cells x hidden_dim

        # Iterate over each cell
        for cell in range(x.shape[1] * x.shape[2]):
            
            # Store hidden state from last layer in H_local at cell location
            H_local[:, cell, :] = self._rnn(self._drop(self._embed(x[:, cell // x.shape[2], cell % x.shape[2], :])))[1][-1] # batch x cell x hidden_dim


        # Sum H_local along cell dimension and for each location subtract HS for that cell to return
        return H_local.sum(dim=1, keepdim=True) - H_local
    
    # Forward function
    def forward(self, x):

        # Global hidden states containing info around current cell already on gpu
        # Across batches for each cell
        H_global = self.cell_hs(x) # batch_size x cells x hidden_dim

        # Tensor to store the full macro cube of size batch x rows x cols
        S_cube = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), device=x.device)
        

        # Loop through all rows x cols cells
        for cell in range(x.shape[1] * x.shape[2]):
            

            # Concatenate global/local context of cell along first dim batch_size then apply dropout
            S_cube[:, cell // x.shape[2], cell % x.shape[2]] =self._pred(
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
                        dim = 1
                    )
                )
            ).view(-1) 
            
        
        # Delete H_global finally
        del H_global
        
        # Return the final S_cube
        return S_cube
        
                
"""
Expanded version stored here for reference
"""
# import torch
# import torch.nn as nn

# class TestRNN(nn.Module):

#     def __init__(self, hidden_state_dim, rnn_layers, embedding_matrix, dropout_rate=0.0, nonlinearity='relu'):
#         super(TestRNN, self).__init__()

#         # Rows of embed matrix = Each word in the vocabulary
#         self.vocab_size = embedding_matrix.shape[0]  # vocab_size = 34057

#         # Cols of embed matrix = Length of each embedding vector
#         self.embedding_dim = embedding_matrix.shape[1]  # embed_dim = 50

#         # The dimension of the hidden state vector 'h' for each step/token
#         self.hidden_dim = hidden_state_dim  # hid_dim = 100

#         # Number of recurrent layers we will use
#         self.rnn_layers = rnn_layers  # rnn_layers = 2

#         # Creates an embedding layer from the pre-trained embedding matrix that maps input tokens to their corresponding word vectors
#         # If freezing then embeddings don't change during training, we need False because we need them to finetune to our task
#         self._embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

#         # Randomly zeroes out a percentage of input units determined by dropout_rate for each update during training
#         self._drop = nn.Dropout(dropout_rate)

#         # RNN layer with 'relu' nonlinearity but not managing exploding gradients, dropout and multiple recurrent layers
#         self._rnn = nn.RNN(
#             self.embedding_dim,
#             self.hidden_dim,
#             self.rnn_layers,
#             nonlinearity=nonlinearity,
#             dropout=dropout_rate,
#             batch_first=True
#         )

#         # Linear layer to map the concatenated hidden states to logits (1 to predict bold or not)
#         self._pred = nn.Linear(2 * self.hidden_dim, 1)
    
#     def cell_hs(self, x):
#         # Set the manual seed for reproducibility
#         torch.manual_seed(0)

#         # Initialize H_local as a zero tensor with the appropriate shape (num_cells, hidden_dim)
#         H_local = torch.zeros(x.shape[0], x.shape[1] * x.shape[2], self.hidden_dim, device=x.device) # batch x cells x hidden_dim
#         H_local_sum = torch.zeros(x.shape[0], self.hidden_dim, device=x.device) # batch x hidden_dim

#         # DEBUG PRINT
#         print(f'\nInput x [batch x rows x cols x tokens]: {x.shape}')
#         print(f'\nH_local before [batch x cells x hidden]: {H_local.shape}')
#         print(f'\nH_local_sum [batch x hidden]: {H_local_sum.shape}')
        
#         # Iterate over each cell
#         for cell in range(x.shape[1] * x.shape[2]):
#             # Get the current row and col
#             row = cell // x.shape[2]
#             col = cell % x.shape[2]
            
#             # Extract cell tokens across batches for current cell
#             celltoks_across_batch = x[:, row, col, :] # batch_size x tokens
            
#             # Get tokens in embedding dim and apply dropout
#             embedded_toks = self._drop(self._embed(celltoks_across_batch)) # batch_size x tokens x embed_dim
            
#             # Now run RNN on dropout input 
#             _, h = self._rnn(embedded_toks) # rnn_layer x batch_size x hidden_dim

#             # Get the last element
#             h_last = h[-1]
            
#             # Store hidden state from last layer in H_local at cell location
#             H_local[:, cell, :] = h_last # batch x cell x hidden_dim
            
#             # Accumulate h_last into H_local_sum
#             H_local_sum += h_last # batch x hidden_dim

#             # DEBUG PRINT for the first cell only
#             if cell == 0:
#                 print(f'\nInside Cell {cell}\nRow {row}, Col {col}')
#                 print(f'\nCell Across [batch x tokens]: {celltoks_across_batch.shape}')
#                 print(f'\nCell Embedded Toks [batch x tokens x embed]: {embedded_toks.shape}')
#                 print(f'\nRNN H [rnn_layers x batch x hidden]: {h.shape}')
#                 print(f'\nLast Layer H [batch x hidden]: {h_last.shape}')
#                 print(f'\nH_local_sum after Cell {cell} [batch x hidden]: {H_local_sum.shape}')
                
#             # Delete intermediate tensors to free up memory
#             del celltoks_across_batch
#             del embedded_toks
#             del h

#         # Calculate ans as the subtraction of each cell's HS in H_local from H_local_sum
#         ans = H_local_sum.unsqueeze(1) - H_local # batch x cells x hidden_dim

#         # DEBUG PRINT
#         print(f'\nFinal answer [batch x cells x hidden]: {ans.shape}')
        
#         return ans

#     # Forward function
#     def forward(self, x):
        
#         # Set the manual seed
#         torch.manual_seed(0)

#         # Global hidden states containing info around current cell already on gpu
#         # Across batches for each cell
#         H_global = self.cell_hs(x) # batch_size x cells x hidden_dim

#         # Tensor to store the full macro cube of size batch x rows x cols
#         S_cube = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), device=x.device)
        
#         # DEBUG PRINT
#         print(f'\nInput x {x.shape}')
#         print(f'\nInitial H_global [batch x cells x hidden]: {H_global.shape}')
#         print(f'\nInitial S_cube [batch x row x col]: {S_cube.shape}')

#         # Loop through all rows x cols cells
#         for cell in tqdm(range(x.shape[1] * x.shape[2]),desc="Doing forward"):
            
#             # Extract global context around cell
#             H_cell = H_global[:, cell, :]
            
#             # Get the current row and col
#             row = cell // x.shape[2]
#             col = cell % x.shape[2]
            
#             # Extract cell tokens across batches for current cell
#             celltoks_across_batch = x[:, row, col, :] # batch_size x tokens
            
#             # Get tokens in embedding dim and apply dropout
#             embedded_toks = self._drop(self._embed(celltoks_across_batch)) # batch_size x tokens x embed_dim

#             # Get output feature from last rnn layer for each token 
#             z, _ = self._rnn(embedded_toks) # batch_size x tokens x hidden_dim
            
#             # Get z for last token across all batches and hidden dim
#             z_lasttok = z[:, -1, :] # batch_size x hidden_dim

#             # Concatenate global/local context of cell along first dim batch_size then apply dropout
#             concat_hs = self._drop(torch.cat((z_lasttok, H_cell), dim = 1)) # batch_size x (2 * hidden_dim)
            
#             # Reshaped fit into S_cube
#             S_cube[:, row, col] = self._pred(concat_hs).view(-1) # batch_size
            
# #             # DEBUG PRINT
# #             if cell == 0:
# #                 print(f'\nInside Cell {cell}\nRow {row}, Col {col}')
# #                 print(f'\nCell Across: {celltoks_across_batch.shape}')
# #                 print(f'\nCell Embedded Toks: {embedded_toks.shape}')
# #                 print(f'\nRNN Z: {z.shape}')
# #                 print(f'\nRNN H: {h.shape}')
# #                 print(f'\nH_cell global HS for cell: {H_cell.shape}')
# #                 print(f'\nRNN Z Last Token: {z_lasttok.shape}')
# #                 print(f'\nConcatenated HS: {concat_hs.shape}')
# #                 print(f'\nPredictions: {preds.shape}\n preds')
# #                 print(f'\nPredictions Reshaped: {S_cube[:, row, col].shape}:\n{S_cube[:, row, col]}')


#             # Delete intermediate tensors to free up memory
#             del celltoks_across_batch
#             del embedded_toks
#             del z
#             del z_lasttok
#             del H_cell
#             del concat_hs
        
        
# #         # DEBUG PRINT
# #         print(f'\nFinal S_cube {S_cube.shape}:\n{S_cube}')
        
#         # Delete H_global finally
#         del H_global
        
#         # Return the final S_cube
#         return S_cube


        
                
# # # Create a DataLoader from your check_loader
# # test_loader = torch.utils.data.DataLoader(check_loader, batch_size=5, shuffle=False)

# # # Get one batch from the DataLoader
# # batch = next(iter(test_loader))

# # # Move the extracted x_tok to gpu
# # exfile = to_gpu(batch['x_tok'],1)

# # # Define a new neural network model to be trained and transfer it to GPU
# # hidden_state_dim = 100
# # rnn_layers = 2
# # rnn_model = to_gpu(TestRNN2(hidden_state_dim, rnn_layers, spreadsheet_wvs),1)

# # # Observe the model
# # rnn_model

# # out = rnn_model.forward(exfile)

# # # Print the shape of S_cube
# # print("S_cube shape:", out.shape)