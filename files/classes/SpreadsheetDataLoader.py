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
    DataLoader class for processing and loading spreadsheet files with vocabulary-based tokenization.

    This class supports processing of spreadsheets in .xls, .xlsx, and .csv formats.
    The processed data includes tokenized values using a vocabulary object and associated
    metadata such as cell type, formatting, and other attributes. Maintains empty x_masks
    list for compatibility with BERT-based processing.

    Attributes:
        vocab: Vocabulary object for encoding tokens from spreadsheet data
        max_rows (int): Maximum number of rows to process per spreadsheet. Defaults to 100.
        max_cols (int): Maximum number of columns to process per spreadsheet. Defaults to 100.
        pad_length (int): Length to which each tokenized cell value is padded. Defaults to 32.
        x_tok (List[torch.LongTensor]): List of 3D tensors containing tokenized values
        x_masks (List): Empty list maintained for compatibility with BERT processing
        y_tok (List[torch.LongTensor]): List of 3D tensors containing metadata for each cell
        file_paths (List[str]): List of file paths for successfully processed spreadsheets
        failed_files (List[str]): List of file paths that failed during processing
    """

    def __init__(self, file_paths: List[str], vocab, max_rows=100, max_cols=100, pad_length=32, threads=4):
        """
        Initializes the SpreadsheetDataLoader and processes all input files.

        Args:
            file_paths (List[str]): List of paths to spreadsheet files
            vocab: Vocabulary object used for token encoding
            max_rows (int, optional): Maximum rows to process. Defaults to 100.
            max_cols (int, optional): Maximum columns to process. Defaults to 100.
            pad_length (int, optional): Padding length for tokenized sequences. Defaults to 32.
            threads (int, optional): Number of parallel processing threads. Defaults to 4.
        """
        # Store parameters
        self.vocab = vocab
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.pad_length = pad_length

        # Initialize data lists
        self.x_tok = []
        self.x_masks = []  # Empty list for compatibility
        self.y_tok = []
        self.file_paths = []
        self.failed_files = []

        # Parallel processing
        results = Parallel(n_jobs=threads, timeout=99999)(
            delayed(self._featurize)(file) 
            for file in tqdm(file_paths, desc="Processing files")
        )

        # Process results
        for x_tok, x_masks, y_tok, path in results:
            if x_tok is not None and y_tok is not None:
                self.x_tok.append(x_tok)
                self.y_tok.append(y_tok)
                self.file_paths.append(path)
            else:
                self.failed_files.append(path)

        print(f'\n{len(self.file_paths) + len(self.failed_files)}(P) = {len(self.file_paths)}(G) + {len(self.failed_files)}(E)')

    def __len__(self):
        """
        Returns the number of successfully processed spreadsheets in the dataset.

        Returns:
            int: Number of spreadsheets in dataset
        """
        return len(self.x_tok)

    def __getitem__(self, index):
        """
        Retrieves data for a single spreadsheet at the specified index.

        Args:
            index (int): Index of spreadsheet to retrieve

        Returns:
            dict: Dictionary containing:
                - x_tok: Tensor of tokenized values for the spreadsheet
                - x_masks: Empty list (maintained for compatibility)
                - y_tok: Tensor of metadata for the spreadsheet
                - file_paths: Path to the source spreadsheet file
        """
        return {
            'x_tok': self.x_tok[index],
            'x_masks': self.x_masks,
            'y_tok': self.y_tok[index],
            'file_paths': self.file_paths[index]
        }

    def _featurize(self, file_path):
        """
        Processes a single spreadsheet file using vocabulary-based tokenization.

        Args:
            file_path (str): Path to spreadsheet file to process

        Returns:
            tuple: Contains:
                - x_tok: Tensor of tokenized values if successful, None if failed
                - x_masks: Empty tensor for compatibility
                - y_tok: Tensor of metadata if successful, None if failed
                - file_path: Path to the processed file
        """
        try:
            x_tok, x_masks, y_tok = process_spreadsheet(
                file_path, 
                max_rows=self.max_rows, 
                max_cols=self.max_cols,
                pad_length=self.pad_length,
                vocab=self.vocab
            )
            return x_tok, x_masks, y_tok, file_path
        except Exception as e:
            return None, None, None, file_path
