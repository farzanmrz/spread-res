# Import and reload the processing func
import importlib
from utils import parseutil
importlib.reload(parseutil)
from utils.parseutil import process_spreadsheet

import torch
from tqdm import tqdm
from typing import List, Optional, Tuple, Any
from joblib import Parallel, delayed
import os


class Loader(torch.utils.data.Dataset):
    """Base loader class for processing spreadsheet data into tensors.

    Handles parallel processing of files and maintains common data structures
    for both vocabulary-based and BERT-based approaches.

    Attributes:
        max_rows (int): Maximum number of rows to process in each spreadsheet.
        max_cols (int): Maximum number of columns to process in each spreadsheet.
        pad_length (int): Length to pad tokenized inputs.
        x_tok (List): List of tokenized inputs.
        x_masks (List): List of attention masks.
        y_tok (List): List of metadata tensors.
        file_paths (List[str]): List of processed file paths.
        failed_files (List[str]): List of file paths that failed processing.
    """

    def __init__(self, file_paths, max_rows=100, max_cols=100, pad_length=32, threads=4):
        """Initialize loader with common parameters.

        Args:
            file_paths (List[str]): List of file paths to process.
            max_rows (int): Maximum number of rows to process in each spreadsheet.
            max_cols (int): Maximum number of columns to process in each spreadsheet.
            pad_length (int): Length to pad tokenized inputs.
            threads (int): Number of threads for parallel processing.
        """
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.pad_length = pad_length

        self.x_tok = []
        self.x_masks = []
        self.y_tok = []
        self.file_paths = []
        self.failed_files = []

        results = Parallel(n_jobs=max(1, threads), timeout=99999)(delayed(self.featurize)(f) for f in tqdm(file_paths, desc="Processing files"))

        for x_tok, x_masks, y_tok, path in results:
            if self.is_valid(x_tok, x_masks, y_tok):
                self.x_tok.append(x_tok)
                self.x_masks.append(x_masks)
                self.y_tok.append(y_tok)
                self.file_paths.append(path)
            else:
                self.failed_files.append(path)

        print(f'\n{len(self.file_paths) + len(self.failed_files)}(P) = {len(self.file_paths)}(G) + {len(self.failed_files)}(E)')

    def is_valid(self, x_tok, x_masks, y_tok):
        """Check if processed results are valid.

        To be implemented by subclasses based on their specific requirements.

        Args:
            x_tok (Any): Tokenized input.
            x_masks (Any): Attention mask.
            y_tok (Any): Metadata tensor.

        Returns:
            bool: True if results are valid, False otherwise.
        """
        raise NotImplementedError

    def featurize(self, file_path):
        """Process a single file into tensors.

        To be implemented by subclasses with specific tokenization logic.

        Args:
            file_path (str): Path to the file to be processed.

        Returns:
            Tuple[Optional[torch.Tensor], ...]: Processed tensors.
        """
        raise NotImplementedError

    def __len__(self):
        """Return number of successfully processed files.

        Returns:
            int: Number of processed files.
        """
        return len(self.x_tok)

    def __getitem__(self, index):
        """Get processed tensors for a single spreadsheet.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing processed tensors and file path.
        """
        return {
            'x_tok': self.x_tok[index], 'x_masks': self.x_masks[index], 'y_tok': self.y_tok[index], 'file_paths': self.file_paths[index]
        }


class LoaderSimple(Loader):
    """Vocabulary-based spreadsheet loader."""

    def __init__(self, file_paths, vocab, **kwargs):
        """Initialize with vocabulary tokenizer.

        Args:
            file_paths (List[str]): List of file paths to process.
            vocab: Vocabulary for tokenization.
            **kwargs: Additional keyword arguments for the base Loader.
        """
        self.vocab = vocab
        super().__init__(file_paths, **kwargs)

    def is_valid(self, x_tok, x_masks, y_tok):
        """Check if tokenization and metadata extraction succeeded.

        Args:
            x_tok (Any): Tokenized input.
            x_masks (Any): Attention mask.
            y_tok (Any): Metadata tensor.

        Returns:
            bool: True if tokenization and metadata extraction succeeded, False otherwise.
        """
        return x_tok is not None and y_tok is not None

    def featurize(self, file_path):
        """Process spreadsheet using vocabulary-based tokenization.

        Args:
            file_path (str): Path to the file to be processed.

        Returns:
            Tuple[Optional[torch.Tensor], ...]: Processed tensors and file path.
        """
        try:
            x_tok, x_masks, y_tok = process_spreadsheet(file_path, max_rows=self.max_rows, max_cols=self.max_cols, pad_length=self.pad_length, vocab=self.vocab)
            return x_tok, x_masks, y_tok, file_path
        except Exception:
            return None, None, None, file_path


class LoaderBert(Loader):
    """BERT-based spreadsheet loader with HuggingFace tokenizer."""

    def __init__(self, file_paths, tokenizer, **kwargs):
        """Initialize with BERT tokenizer and disable parallel tokenization.

        Args:
            file_paths (List[str]): List of file paths to process.
            tokenizer: BERT tokenizer.
            **kwargs: Additional keyword arguments for the base Loader.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = tokenizer
        super().__init__(file_paths, **kwargs)

    def is_valid(self, x_tok, x_masks, y_tok):
        """Verify all tensors were generated successfully.

        Args:
            x_tok (Any): Tokenized input.
            x_masks (Any): Attention mask.
            y_tok (Any): Metadata tensor.

        Returns:
            bool: True if all tensors were generated successfully, False otherwise.
        """
        return all(x is not None for x in (x_tok, x_masks, y_tok))

    def featurize(self, file_path):
        """Process spreadsheet using BERT tokenization.

        Args:
            file_path (str): Path to the file to be processed.

        Returns:
            Tuple[Optional[torch.Tensor], ...]: Processed tensors and file path.
        """
        try:
            x_tok, x_masks, y_tok = process_spreadsheet(file_path, max_rows=self.max_rows, max_cols=self.max_cols, pad_length=self.pad_length, tokenizer=self.tokenizer)
            return x_tok, x_masks, y_tok, file_path
        except Exception:
            return None, None, None, file_path