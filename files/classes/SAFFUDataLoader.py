# Imports for parsing file
import importlib
from utils import parsesaffu
importlib.reload(parsesaffu)

from utils.parsesaffu import process_spreadsheet
import torch
from tqdm import tqdm
import os
from typing import List
from joblib import Parallel, delayed

class SAFFUDataLoader(torch.utils.data.Dataset):
	"""
	SAFFU DataLoader class for processing and loading spreadsheet files into tokenized train (x_tok)
	and associated metadata (y_tok) for machine learning purposes.

	This class supports processing of spreadsheets in `.xls`, `.xlsx`, and `.csv` formats.
	The processed train includes tokenized values of the spreadsheet content and the corresponding
	metadata such as cell type, formatting, and other attributes. The train is loaded into PyTorch
	tensors, making it compatible for training models that require spreadsheet train as input.

	Attributes:
		tokenizer: The tokenizer object used to encode tokens from the spreadsheet train.
		max_rows (int): The maximum number of rows to process per spreadsheet. Defaults to 100.
		max_cols (int): The maximum number of columns to process per spreadsheet. Defaults to 100.
		pad_length (int): The length to which each tokenized cell value is padded. Defaults to 32.
		x_tok (List[torch.LongTensor]): A list of 3D PyTorch tensors containing tokenized train for each spreadsheet.
		y_tok (List[torch.LongTensor]): A list of 3D PyTorch tensors containing metadata for each spreadsheet.
		file_paths (List[str]): A list of file paths corresponding to the successfully processed spreadsheets.
	"""

	def __init__( self, file_paths: List[ str ], tokenizer, max_rows = 100, max_cols = 100, pad_length = 32 ):
		"""
		Initializes the SpreadsheetDataLoader with file paths, vocabulary, and padding length.

		Args:
			file_paths (List[str]): List of file paths to the spreadsheets.
			tokenizer: Tokenizer object for encoding tokens.
			max_rows (int, optional): Maximum number of rows to process per spreadsheet. Defaults to 100.
			max_cols (int, optional): Maximum number of columns to process per spreadsheet. Defaults to 100.
			pad_length (int, optional): Padding length for tokenized train. Defaults to 32.
		"""
		# Store the vocabulary, rows, cols and padding length
		self.tokenizer = tokenizer
		self.max_rows = max_rows
		self.max_cols = max_cols
		self.pad_length = pad_length

		# Initialize lists to hold the train tensors (x_tok), metadata (y_tok), and successfully processed file paths
		self.x_tok = [ ]
		self.y_tok = [ ]
		self.file_paths = [ ]

		# Go through all the file paths with tqdm bar
		for file in tqdm(file_paths, desc = "Processing Files"):

			# Run the featurize function on each file
			x, y, p = self._featurize(file)

			# If we receive None then skip this file
			if x is None or y is None or p is None:
				continue

			# Append the tensors and paths to relevant place
			self.x_tok.append(x)
			self.y_tok.append(y)
			self.file_paths.append(p)


	def __len__( self ):
		"""
		Returns the number of samples in the dataset.

		Returns:
			int: Number of samples (i.e., number of processed spreadsheets).
		"""
		return len(self.x_tok)

	def __getitem__( self, index ):
		"""
		Retrieves the train tensors and metadata for the given index.

		Args:
			index (int): Index of the sample to retrieve.

		Returns:
			dict: Dictionary containing 'x_tok' and 'y_tok' corresponding to the given index.
		"""
		return { 'x_tok': self.x_tok[ index ], 'y_tok': self.y_tok[ index ] }

	def _featurize( self, file_path ):
		"""
		Processes a single spreadsheet file and returns the tokenized train, metadata, and file path.

		Args:
			file_path (str): The path to the spreadsheet file.

		Returns:
			tuple: A tuple containing the tokenized train (x_tok), metadata (y_tok),
				   and file path (str) if successful; otherwise, (None, None, None).
		"""

		# Try catch block to avoid errant files
		try:

			# Get the tokenized data and metadata
			result = process_spreadsheet(file_path, self.tokenizer, self.max_rows, self.max_cols, self.pad_length)

			# Func returns None in case of error, if not none
			if result is not None:

				# Return the x, y tensors and the file path
				return result[ 0 ], result[ 1 ], file_path

			# Otherwise if None then just return None
			else:
				return None, None, None

		# Except block for exception return None
		except Exception as e:
			return None, None, None


# # Imports for parsing file
# import importlib
# from utils import parsesaffu
# importlib.reload(parsesaffu)

# from utils.parsesaffu import process_spreadsheet
# import torch
# from tqdm import tqdm
# import os
# from typing import List
# from joblib import Parallel, delayed

# class SAFFUDataLoader(torch.utils.data.Dataset):
# 	"""
# 	SAFFU DataLoader class for processing and loading spreadsheet files into tokenized train (x_tok)
# 	and associated metadata (y_tok) for machine learning purposes.

# 	This class supports processing of spreadsheets in `.xls`, `.xlsx`, and `.csv` formats.
# 	The processed train includes tokenized values of the spreadsheet content and the corresponding
# 	metadata such as cell type, formatting, and other attributes. The train is loaded into PyTorch
# 	tensors, making it compatible for training models that require spreadsheet train as input.

# 	Attributes:
# 		tokenizer: The tokenizer object used to encode tokens from the spreadsheet train.
# 		max_rows (int): The maximum number of rows to process per spreadsheet. Defaults to 100.
# 		max_cols (int): The maximum number of columns to process per spreadsheet. Defaults to 100.
# 		pad_length (int): The length to which each tokenized cell value is padded. Defaults to 32.
# 		x_tok (List[torch.LongTensor]): A list of 3D PyTorch tensors containing tokenized train for each spreadsheet.
# 		y_tok (List[torch.LongTensor]): A list of 3D PyTorch tensors containing metadata for each spreadsheet.
# 		file_paths (List[str]): A list of file paths corresponding to the successfully processed spreadsheets.
# 	"""

# 	def __init__( self, file_paths: List[ str ], tokenizer, max_rows = 100, max_cols = 100, pad_length = 32 ):
# 		"""
# 		Initializes the SpreadsheetDataLoader with file paths, vocabulary, and padding length.

# 		Args:
# 			file_paths (List[str]): List of file paths to the spreadsheets.
# 			tokenizer: Tokenizer object for encoding tokens.
# 			max_rows (int, optional): Maximum number of rows to process per spreadsheet. Defaults to 100.
# 			max_cols (int, optional): Maximum number of columns to process per spreadsheet. Defaults to 100.
# 			pad_length (int, optional): Padding length for tokenized train. Defaults to 32.
# 		"""
# 		# Store the vocabulary, rows, cols and padding length
# 		self.tokenizer = tokenizer
# 		self.max_rows = max_rows
# 		self.max_cols = max_cols
# 		self.pad_length = pad_length

# 		# Initialize lists to hold the train tensors (x_tok), metadata (y_tok), and successfully processed file paths
# 		self.x_tok = [ ]
# 		self.y_tok = [ ]
# 		self.file_paths = [ ]

# 		# Parallel processing with tqdm progress bar
# 		results = Parallel(n_jobs=os.cpu_count() // 2)(
# 			delayed(self._featurize)(file) for file in tqdm(file_paths, desc="Processing Files")
# 		);print('Done, zipping')

# 		# Filter and append valid results in a single comprehension
# 		self.x_tok, self.y_tok, self.file_paths = zip(*[(x, y, p) for x, y, p in results if x is not None])

# 	def __len__( self ):
# 		"""
# 		Returns the number of samples in the dataset.

# 		Returns:
# 			int: Number of samples (i.e., number of processed spreadsheets).
# 		"""
# 		return len(self.x_tok)

# 	def __getitem__( self, index ):
# 		"""
# 		Retrieves the train tensors and metadata for the given index.

# 		Args:
# 			index (int): Index of the sample to retrieve.

# 		Returns:
# 			dict: Dictionary containing 'x_tok' and 'y_tok' corresponding to the given index.
# 		"""
# 		return { 'x_tok': self.x_tok[ index ], 'y_tok': self.y_tok[ index ] }

# 	def _featurize( self, file_path ):
# 		"""
# 		Processes a single spreadsheet file and returns the tokenized train, metadata, and file path.

# 		Args:
# 			file_path (str): The path to the spreadsheet file.

# 		Returns:
# 			tuple: A tuple containing the tokenized train (x_tok), metadata (y_tok),
# 				   and file path (str) if successful; otherwise, (None, None, None).
# 		"""

# 		# Try catch block to avoid errant files
# 		try:
# 			# Get the tokenized data and metadata
# 			result = process_spreadsheet(file_path, self.tokenizer, self.max_rows, self.max_cols, self.pad_length)

# 			# Func returns None in case of error, if not none
# 			if result is not None:
# 				# Return the x, y tensors and the file path
# 				return result[ 0 ], result[ 1 ], file_path
# 			else:
# 				return None
# 		except Exception as e:
# 			return None
