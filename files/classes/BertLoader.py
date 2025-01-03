# Import parsing function in the required order
import importlib
from utils import parseutil

importlib.reload(parseutil)
from utils.parseutil import process_spreadsheet

# Other imports
import os
import torch
from tqdm import tqdm
from joblib import Parallel, delayed

# Disable tokenizer parallelization to avoid errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BertLoader(torch.utils.data.Dataset):
    """
    DataLoader class for processing and loading spreadsheet files into tokenized inputs (x_tok),
    attention masks (x_masks), and associated metadata (y_tok) for machine learning purposes.

    This class supports processing of spreadsheets in `.xls`, `.xlsx`, and `.csv` formats.
    The processed data includes:
      - input_ids (x_tok),
      - attention masks (x_masks),
      - metadata (y_tok) such as cell type, formatting, and other attributes.

    All three are loaded into PyTorch tensors, making it compatible for training models that require
    spreadsheet data as input.

    Attributes:
        tokenizer: The Hugging Face tokenizer used to encode tokens from each cell value.
        max_rows (int): The maximum number of rows to process per spreadsheet. Defaults to 100.
        max_cols (int): The maximum number of columns to process per spreadsheet. Defaults to 100.
        pad_length (int): The sequence length (padding/truncation) for each cell. Defaults to 32.
        x_tok (list[torch.LongTensor]): A list of 3D PyTorch tensors (each shape = [max_rows, max_cols, pad_length])
                                        containing input_ids per spreadsheet.
        x_masks (list[torch.LongTensor]): A list of 3D PyTorch tensors (same shape) for attention masks.
        y_tok (list[torch.LongTensor]): A list of 3D PyTorch tensors (shape = [max_rows, max_cols, 17])
                                        containing metadata for each spreadsheet.
        file_paths (list[str]): A list of file paths corresponding to the successfully processed spreadsheets.
        failed_files (list[str]): A list of file paths that encountered errors during processing.
    """

    def __init__(
        self,
        file_paths,  
        tokenizer,  # A Hugging Face tokenizer
        max_rows=100,
        max_cols=100,
        pad_length=32,
        threads=4,
    ):
        """
        Initializes the BertLoader with file paths, tokenizer, rows/cols, and pad length.

        Args:
            file_paths: List of file paths to the spreadsheets (no explicit type annotation).
            tokenizer: A Hugging Face tokenizer object.
            max_rows (int, optional): Maximum number of rows to process per spreadsheet. Defaults to 100.
            max_cols (int, optional): Maximum number of columns to process per spreadsheet. Defaults to 100.
            pad_length (int, optional): The sequence length (padding/truncation). Defaults to 32.
            threads (int, optional): Number of parallel threads (via joblib). Defaults to 4.
        """
        # Store relevant parameters
        self.tokenizer = tokenizer
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.pad_length = pad_length

        # Lists to hold the data
        self.x_tok = []
        self.x_masks = []
        self.y_tok = []

        # Lists to hold successful and failed file paths
        self.file_paths = []
        self.failed_files = []

        # Parallel processing of files
        results = Parallel(n_jobs=max(1, threads), timeout=99999)(
            delayed(self._featurize)(f)
            for f in tqdm(file_paths, desc="Processing files")
        )

        # Collect results
        for x_tok_cur, x_masks_cur, y_tok_cur, path in results:
            # Check if x_tok_cur, x_masks_cur, and y_tok_cur are not None => success
            if (
                x_tok_cur is not None
                and x_masks_cur is not None
                and y_tok_cur is not None
            ):
                self.x_tok.append(x_tok_cur)
                self.x_masks.append(x_masks_cur)
                self.y_tok.append(y_tok_cur)
                self.file_paths.append(path)
            else:
                # Otherwise it's a failed file
                self.failed_files.append(path)

        # Log summary
        print(
            f"\n{len(self.file_paths) + len(self.failed_files)}(P) = "
            f"{len(self.file_paths)}(G) + {len(self.failed_files)}(E)"
        )

    def __len__(self):
        """
        Returns the number of samples in the dataset (i.e., number of processed spreadsheets).
        """
        return len(self.x_tok)

    def __getitem__(self, index):
        """
        Retrieves the data (x_tok, x_masks, y_tok) for the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'x_tok', 'x_masks', 'y_tok',
                  and 'file_paths' for the requested spreadsheet.
        """
        return {
            "x_tok": self.x_tok[index],
            "x_masks": self.x_masks[index],
            "y_tok": self.y_tok[index],
            "file_paths": self.file_paths[index],
        }

    def _featurize(self, file_path):
        """
        Processes a single spreadsheet file and returns (x_tok, x_masks, y_tok, file_path).

        Args:
            file_path (str): The path to the spreadsheet file.

        Returns:
            tuple: (x_tok, x_masks, y_tok, file_path) if successful,
                   otherwise (None, None, None, file_path).
        """
        try:
            # process_spreadsheet now returns (x_tok, x_masks, y_tok)
            x_tok_cur, x_masks_cur, y_tok_cur = process_spreadsheet(
                file_path, self.max_rows, self.max_cols, self.pad_length, tokenizer=self.tokenizer
            )
            return x_tok_cur, x_masks_cur, y_tok_cur, file_path

        except Exception:
            # On exception, return None for the data but keep the file path
            return None, None, None, file_path
