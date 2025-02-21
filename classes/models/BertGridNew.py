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
