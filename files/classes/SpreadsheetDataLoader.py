import importlib
from utils import parseutil
importlib.reload(parseutil)
from utils.parseutil import process_spreadsheet

import torch
from tqdm import tqdm
from typing import List
from joblib import Parallel, delayed
import os


class SpreadsheetDataLoader(torch.utils.data.Dataset):
    """
    DataLoader class for processing and loading spreadsheet files into tokenized train (x_tok)
    and associated metadata (y_tok) for machine learning purposes.

    This class supports processing of spreadsheets in `.xls`, `.xlsx`, and `.csv` formats.
    The processed train includes tokenized values of the spreadsheet content and the corresponding
    metadata such as cell type, formatting, and other attributes. The train is loaded into PyTorch
    tensors, making it compatible for training models that require spreadsheet train as input.

    Attributes:
        vocab: The vocabulary object used to encode tokens from the spreadsheet train.
        max_rows (int): The maximum number of rows to process per spreadsheet. Defaults to 100.
        max_cols (int): The maximum number of columns to process per spreadsheet. Defaults to 100.
        pad_length (int): The length to which each tokenized cell value is padded. Defaults to 32.
        x_tok (List[torch.LongTensor]): A list of 3D PyTorch tensors containing tokenized train for each spreadsheet.
        y_tok (List[torch.LongTensor]): A list of 3D PyTorch tensors containing metadata for each spreadsheet.
        file_paths (List[str]): A list of file paths corresponding to the successfully processed spreadsheets.
    """

    def __init__( self, file_paths: List[ str ], vocab, max_rows = 100, max_cols = 100, pad_length = 32, threads = 4 ):
        """
        Initializes the SpreadsheetDataLoader with file paths, vocabulary, and padding length.

        Args:
            file_paths (List[str]): List of file paths to the spreadsheets.
            vocab: Vocabulary object for encoding tokens.
            max_rows (int, optional): Maximum number of rows to process per spreadsheet. Defaults to 100.
            max_cols (int, optional): Maximum number of columns to process per spreadsheet. Defaults to 100.
            pad_length (int, optional): Padding length for tokenized train. Defaults to 32.
        """
        # Store the vocabulary, rows, cols and padding length
        self.vocab = vocab
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.pad_length = pad_length

        # Initialize lists to hold the train tensors (x_tok), metadata (y_tok)
        self.x_tok = [ ]
        self.y_tok = [ ]

        # List to hold the successful/failed file paths
        self.file_paths = [ ]
        self.failed_files = [ ]

        # Parallel processing of files using joblib
        results = Parallel(n_jobs = threads, timeout=99999)(delayed(self._featurize)(file) for file in tqdm(file_paths, desc = "Processing files"))


        # Loop through all 3 vars of results
        for x_tok, y_tok, path in results:

            # Check if x_tok and y_tok are not None for successful files
            if x_tok is not None and y_tok is not None:
                self.x_tok.append(x_tok)
                self.y_tok.append(y_tok)
                self.file_paths.append(path)

            # If either of them is None then append the file path to the failed files list
            else:
                self.failed_files.append(path)

        # Log summary
        print(f'\n{len(self.file_paths) + len(self.failed_files)}(P) = {len(self.file_paths)}(G) + {len(self.failed_files)}(E)')


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
        return { 
          'x_tok': self.x_tok[ index ], 
          'y_tok': self.y_tok[ index ],
          'file_paths': self.file_paths[ index ]
           }

    def _featurize( self, file_path ):
        """
        Processes a single spreadsheet file and returns the tokenized train, metadata, and file path.

        Args:
            file_path (str): The path to the spreadsheet file.

        Returns:
            tuple: A tuple containing the tokenized train (x_tok), metadata (y_tok),
                   and file path (str) if successful; otherwise, (None, None, None).
        """

        # Process the spreadsheet the normal way to get x_tok and y_tok and return with file path
        try:
            x_tok_curr, y_tok_curr = process_spreadsheet(file_path, self.vocab, self.max_rows, self.max_cols, self.pad_length)
            return x_tok_curr, y_tok_curr, file_path

        # In case of exception return None and the file path
        except Exception as e:
            return None, None, file_path
