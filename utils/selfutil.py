# Import then Class Reset
import importlib
from classes import Vocab

importlib.reload(Vocab)
from classes.Vocab import Vocab

# Other imports
import pandas as pd
import warnings
import gensim.downloader as api
import torch
import re
import random
import numpy as np
import gc
from joblib import Parallel, delayed
from collections import Counter
from tqdm import tqdm
import os
import sys
import csv
import chardet


def set_seed(seed: int):
    """
    Set seed for reproducibility across CPU, CUDA, and MPS devices.

    Args:
        seed (int): The seed value to set.
    """
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy's seed
    np.random.seed(seed)

    # Set PyTorch seed for CPU
    torch.manual_seed(seed)

    # Check if CUDA is available and set seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For all GPUs
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = (
            False  # Disable auto-tuning for reproducibility
        )

    # Optional: Ensure reproducibility by controlling environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Print confirmation


def to_gpu(x, device=2, seed=0):
    """
    Transfers the input tensor or model to CUDA if available,
    else remains on CPU. If multiple GPUs are available, it uses DataParallel.
    Also sets the random seed for reproducibility.

    Parameters:
    x (torch.Tensor or nn.Module): The tensor or model to be transferred.
    device (int): The CUDA device index to use. Defaults to 0 (i.e., 'cuda:0').
    seed (int): The seed value to set for reproducibility. Defaults to 0.

    Returns:
    torch.Tensor or nn.Module: The tensor or model transferred to the appropriate device (CUDA or CPU).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Check if the specified device is available
        if device < torch.cuda.device_count():
            x = x.to(f"cuda:{device}")
        else:
            raise RuntimeError(f"CUDA device 'cuda:{device}' is not available.")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        torch.mps.empty_cache()
        x = x.to("mps")

    else:
        x = x.to("cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gc.collect()

    return x


def get_fileList(directory):

    """
    Retrieves and classifies files in a directory into valid and invalid files based on extensions.

    Args:
        directory (str): Path to the directory to scan for files.

    Returns:
        tuple: A tuple containing two lists:
            - valid_files (list): List of file paths with supported extensions (.xls, .xlsx, .csv).
            - invalid_files (list): List of file paths with unsupported extensions.

    Raises:
        ValueError: If the specified directory does not exist.
    """

    # Ensure the provided directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} NOT FOUND")

    # Files with supported extensions
    valid_files = [
        os.path.join(directory, file_name)
        for file_name in os.listdir(directory)
        if file_name.endswith((".xls", ".xlsx", ".csv"))
        and os.path.isfile(os.path.join(directory, file_name))
    ]

    # Files with unsupported extensions
    invalid_files = [
        os.path.join(directory, file_name)
        for file_name in os.listdir(directory)
        if not file_name.endswith((".xls", ".xlsx", ".csv"))
        and os.path.isfile(os.path.join(directory, file_name))
    ]

    # Return both lists
    return valid_files, invalid_files

def tokenize(text, space=True, case="lower"):

    # Declare empty list to store the tokens
    tokens = []

    # Now apply different sorts of pattern matching for tokenization
    for token in re.split("([0-9a-zA-Z'-]+)", text):
        if not space:
            token = re.sub("[ ]+", "", token)
        if not token:
            continue
        if re.search("[0-9a-zA-Z'-]", token):
            tokens.append(token)
        else:
            tokens.extend(token)

    # Adjust the list of tokens based on the specified case
    if case == "lower":
        tokens = [token.lower() for token in tokens]
    elif case == "upper":
        tokens = [token.upper() for token in tokens]

    # Return the list of tokens
    return tokens


def tokenize_pad(cell_value, vocab, pad_length=32):
    """
    Tokenizes, pads, and encodes a cell value.

    Args:
        cell_value (str): The string value of the cell to be processed.
        vocab: Vocabulary object for encoding tokens.
        pad_length (int, optional): The length to which the tokenized list is padded. Defaults to 32.

    Returns:
        List[int]: A list of encoded tokens, padded or truncated to the specified length.
    """
    # Derive case from the vocab object
    case = vocab._case

    # According to case set the cls, eos and pad tokens
    cls_token = vocab.cls_token
    eos_token = vocab.sep_token
    pad_token = vocab.pad_token

    # Pass the cell value along with the case to tokenize
    tokens = tokenize(cell_value, space=vocab._space, case=vocab._case)

    # Add the <cls> and <eos> tokens
    input_tokens = [cls_token] + tokens + [eos_token]

    # Pad or truncate the token list to the specified pad_length
    if len(input_tokens) < pad_length:
        remaining_length = pad_length - len(input_tokens)
        input_tokens.extend([pad_token] * remaining_length)
    else:
        input_tokens = input_tokens[:pad_length]

    # Encode the tokens using the provided vocabulary
    toks_encoded = [vocab.encode(tok) for tok in input_tokens]

    return toks_encoded


def get_vocab(
    data_dir="../../data/farzan/train_big/",
    vocab_size=50000,
    space=True,
    case="lower",
    threads=4,
):
    """
    Collects all .xls, .xlsx, and .csv files from the specified directory,
    processes each file in parallel, counts occurrences of each token based on the specified case,
    and trains a Vocab object.

    Args:
        data_dir (str): Directory containing the files to be processed.
        vocab_size (int): Desired vocabulary size. Trains a Vocab object on `vocab_size - 4` most common tokens.
        case (str): Determines the case for token processing ('lower', 'upper', 'both').
        threads (int): Determines the level of parallelism for token aggregation.

    Returns:
        Vocab: A trained Vocab object.
    """

    # Get valid and invalid file path lists
    file_paths, invalid_files = get_fileList(data_dir)

    # Initialize errant files counter from the length of invalid_files at the start and vocab object
    init_total_files = len(file_paths) + len(invalid_files)
    ec = len(invalid_files)
    vocab = Vocab(target=False, space=space, case=case)


    # Helper function for token aggregation from a single file
    def process_file(file_path):
        try:
            # Tokenize and count tokens for the file
            return Counter(
                [
                    token
                    for cell in spreadsheet_to_df(file_path).values.flatten()
                    if cell is not None
                    for token in tokenize(cell, space=space, case=case)
                ]
            )
        except Exception:
            # Increment error count
            nonlocal ec
            ec += 1
            return Counter()  # Return an empty counter for failed files

    # Process files in parallel and combine results
    aggregated_token_counts = sum(
        Parallel(n_jobs=threads, timeout=99999)(
            delayed(process_file)(file_path)
            for file_path in tqdm(file_paths, desc="Getting Vocab")
        ),
        Counter(),  # Start with an empty counter
    )

    # Train the vocab object
    vocab.train(
        [token for token, count in aggregated_token_counts.most_common(vocab_size - 4)]
        if (vocab_size - 4 < len(aggregated_token_counts))
        else list(aggregated_token_counts.keys())
    )

    # Print statistics
    print(f"{init_total_files}(P) = {init_total_files - ec}(G) + {ec}(E)")
    print(f"Unique Tokens: {len(aggregated_token_counts)}")
    print(f"Vocab Size: {len(vocab._word2idx)}")

    # Return the trained vocab object
    return vocab


def spreadsheet_to_df(file_path, max_rows=100, max_cols=100):
    """
    Reads a spreadsheet file (XLS, XLSX, or CSV) and converts it into a pandas DataFrame with a fixed size.

    Parameters:
    file_path (str): The path to the spreadsheet file.
    max_rows (int, optional): The maximum number of rows for the DataFrame. Defaults to 100.
    max_cols (int, optional): The maximum number of columns for the DataFrame. Defaults to 100.

    Returns:
    pd.DataFrame: A pandas DataFrame with the content of the spreadsheet, resized to max_rows x max_cols.
    """
    # Suppress warnings related to deprecated features in pandas
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="openpyxl.worksheet.header_footer"
    )
    warnings.filterwarnings(
        "ignore", message="Workbook contains no default style, apply openpyxl's default"
    )  # No default styling
    warnings.filterwarnings(
        "ignore", message="Unknown extension is not supported and will be removed"
    )  # Unknown extensions in file
    warnings.filterwarnings(
        "ignore",
        message="Cell .* is marked as a date but the serial value .* is outside the limits for dates.*",
    )  # Errant date format
    warnings.filterwarnings("ignore", category=UserWarning)  # Surpress all UserWarnings

    # Extract the file extension
    extension = file_path.split(".")[-1].lower()

    # Raise error if invalid file
    if extension not in ["xlsx", "xls", "csv"]:
        raise ValueError(f"spreadsheet_to_df ERROR {file_path}-Unsupported extension")

    try:

        # Replace match-case with if-elif for Python 3.8 compatibility
        if extension == "xlsx":
            # Read the Excel file using openpyxl
            df = pd.read_excel(
                file_path,
                header=None,
                dtype=str,
                na_values=" ",
                keep_default_na=False,
                engine="openpyxl",
            )

        elif extension == "xls":
            # Read the Excel file using xlrd
            df = pd.read_excel(
                file_path,
                header=None,
                dtype=str,
                na_values=" ",
                keep_default_na=False,
                engine="xlrd",
            )

        elif extension == "csv":
            # Try reading the file as a CSV
            df = pd.read_csv(
                file_path,
                header=None,
                dtype=str,
                na_values=" ",
                keep_default_na=False,
                sep=None,
                engine="python",
            )

    except Exception as parse_error:
        raise parse_error


    # Fill NaN values with empty strings and convert to string type
    df = df.fillna("").astype(str)

    # Put condition if 0,0 is passed then don't do any editing and return df
    if max_rows == 0 and max_cols == 0:
        return df

    # Ensure the DataFrame is of size max_rows x max_cols in normal case
    df_fixed = df.reindex(index=range(max_rows), columns=range(max_cols), fill_value="")

    # Return the fixed DataFrame
    return df_fixed


# Function to get delimiter from a file, attempting to treat xls as CSV
def get_delimencoding(file_path):

    # Step 1: Get Encoding
    try:

        # Read the entire content in binary mode
        with open(file_path, mode="rb") as binary_file:
            content = binary_file.read()

        # Detect encoding
        curr_encoding = chardet.detect(content)["encoding"]

    # If encoding detection fails, use 'utf-8' as default
    except Exception:
        curr_encoding = "utf-8"

    # Step 2: Get Delimiter
    try:

        # Use csv.Sniffer to detect delimiter
        delimiter = csv.Sniffer().sniff(content.decode(curr_encoding)).delimiter

    # If delimiter detection fails, use '\t' as default
    except Exception:
        delimiter = "\t"

    return delimiter, curr_encoding


def create_embeddings(vocab, model_name="glove-wiki-gigaword-50"):
    """
    Create word vectors using a pre-trained model from gensim.

    Parameters:
    - vocab: The vocabulary object with encode/decode methods.
    - model_name: The name of the pre-trained model to use. Default is "glove-wiki-gigaword-50".

    Returns:
    - torch.Tensor: A tensor of shape (vocab_size, vocab_dim) containing the word vectors.
    """
    # Set the same seed for reproducibility
    torch.manual_seed(0)

    # Load the pre-trained model
    model = api.load(model_name)

    # Determine the dimensions of the word vectors from vocabulary size and vector size of model
    vocab_size = len(vocab._word2idx)
    vocab_dim = model.vector_size

    # Initialize the word vectors as 0s initially
    word_vectors = torch.zeros(vocab_size, vocab_dim)

    # Loop through each word in the vocabulary
    for ix in tqdm(range(vocab_size), desc="Creating Word Embeddings"):

        # Get the actual word from the vocabulary decoded from its corresponding index
        word = vocab.decode(ix)

        # Use the pre-trained vector if word in model, otherwise initialize randomly
        word_vectors[ix, :] = (
            torch.tensor(model[word]) if word in model else torch.randn(vocab_dim)
        )

    # Print statement indicating shape
    print(f"Word Embeddings Shape: {word_vectors.shape}")

    return word_vectors
