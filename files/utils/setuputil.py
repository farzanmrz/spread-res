# General imports
import os
import torch
import importlib
import copy
import json
from transformers import AutoTokenizer

# Reload the selfutil module and import required functions
from utils import selfutil
from classes import SpreadsheetDataLoader, BertLoader
importlib.reload(selfutil)
importlib.reload(SpreadsheetDataLoader)
importlib.reload(BertLoader)
from utils.selfutil import set_seed, get_vocab, create_embeddings, get_fileList
from classes.SpreadsheetDataLoader import SpreadsheetDataLoader
from classes.BertLoader import BertLoader

#################################################################################
## SIMPLE MODEL FUNCS

# Define the display_config function
def display_simple_config(config):
    """
    Display the configuration with non-serializable objects replaced by string placeholders.

    Parameters:
    - config: The configuration dictionary.
    """
    # Create a deep copy of the config to avoid modifying the original
    config_serializable = copy.deepcopy(config)

    # Replace non-serializable objects with string representations
    config_serializable["DEVICE"] = str(config_serializable["DEVICE"])  # Serialize the device object
    config_serializable["vocab"]["object"] = "<Vocab Object>"  # Replace the vocab object with a placeholder
    config_serializable["wvs"] = "<Embedding Tensor>"  # Replace the embeddings tensor with a placeholder

    # Replace dataloaders with placeholders
    config_serializable["train_loader"] = "<Train SpreadsheetDataLoader Object>"
    config_serializable["val_loader"] = "<Validation SpreadsheetDataLoader Object>"
    config_serializable["test_loader"] = "<Test SpreadsheetDataLoader Object>"

    # Pretty-print the configuration
    print("\nFinal configuration:")
    print(json.dumps(config_serializable, indent=2))


# Define the simple_setup function
def setup_simple_config(setup_config):
    # Initialize the config dictionary
    config = {}

    ######## DATA DIR ########

    # Combine and validate data_dir
    if os.path.isdir(setup_config["data_dir"]):
        config["data_dir"] = setup_config["data_dir"]
    else:
        raise ValueError(f"ERR: data_dir '{setup_config['data_dir']}' is not a valid path")

    ######## DEVICE ########

    # Validate and set DEVICE
    if (
        setup_config["device"].startswith("cuda")
        and torch.cuda.is_available()
        and int(setup_config["device"].split(":")[1]) < torch.cuda.device_count()
    ):
        config["DEVICE"] = torch.device(setup_config["device"])
    elif (
        setup_config["device"].startswith("mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        config["DEVICE"] = torch.device("mps")
    else:
        config["DEVICE"] = torch.device("cpu")

    ######## THREADS ########

    # Validate and set THREADS
    remaining_threads = os.cpu_count() - (os.cpu_count() // setup_config["threads"])
    if remaining_threads >= 4:
        config["THREADS"] = setup_config["threads"]
    else:
        raise ValueError(f"ERR: {remaining_threads} threads remaining leave at least 4")

    ######## SEED ########

    # Set the random seed
    set_seed(setup_config["seed"])

    ######## VOCAB ########

    # Combine and validate vocab_dir
    vocab_dir = os.path.join(setup_config["data_dir"], setup_config["vocab"]["ds"])
    if os.path.isdir(vocab_dir):
        config["vocab"] = {"dir": vocab_dir}
    else:
        raise ValueError(f"ERR: vocab_dir '{vocab_dir}' is not a valid folder")

    # Validate and set vocab_size
    if isinstance(setup_config["vocab"]["size"], int) and 4 <= setup_config["vocab"]["size"] <= 2000000:
        config["vocab"]["size"] = setup_config["vocab"]["size"]
    else:
        raise ValueError(f"ERR: vocab size '{setup_config['vocab']['size']}' must be an integer between 4 and 2,000,000")

    # Validate and set vocab_space
    vocab_space = setup_config["vocab"].get("space", True)
    config["vocab"]["space"] = vocab_space if isinstance(vocab_space, bool) else True

    # Validate and set vocab_case
    vocab_case = setup_config["vocab"].get("case", "lower")
    config["vocab"]["case"] = vocab_case if vocab_case in ["both", "upper", "lower"] else "lower"

    # Generate the vocab object using get_vocab
    config["vocab"]["object"] = get_vocab(
        config["vocab"]["dir"],
        config["vocab"]["size"],
        space=config["vocab"]["space"],
        case=config["vocab"]["case"],
        threads=config["THREADS"]
    )

    ######## WVS ########

    # Generate the embedding matrix (wvs) using create_embeddings
    config["wvs"] = create_embeddings(config["vocab"]["object"])

    ######## DATA_DS ########

    # Validate and set train, val, test directories
    train_dir = os.path.join(config["data_dir"], f"{setup_config['data_ds']}_train")
    val_dir = os.path.join(config["data_dir"], f"{setup_config['data_ds']}_val")
    test_dir = os.path.join(config["data_dir"], f"{setup_config['data_ds']}_test")

    missing_dirs = [dir_name for dir_name, path in 
        {"train": train_dir, "val": val_dir, "test": test_dir}.items() 
        if not os.path.isdir(path)]

    if missing_dirs:
        raise ValueError(f"ERR: Missing dataset directories: {', '.join(missing_dirs)}")

    config["train_dir"] = train_dir
    config["val_dir"] = val_dir
    config["test_dir"] = test_dir

    ######## DATALOADERS ########

    # Set max_rows, max_cols, num_tokens
    config["rows"] = setup_config["rows"]
    config["cols"] = setup_config["cols"]
    config["tokens"] = setup_config["tokens"]

    # Generate file lists
    train_files, _ = get_fileList(config["train_dir"])
    val_files, _ = get_fileList(config["val_dir"])
    test_files, _ = get_fileList(config["test_dir"])

    # Create data loaders
    config["train_loader"] = SpreadsheetDataLoader(
        train_files, config["vocab"]["object"], config["rows"], config["cols"], config["tokens"], threads=config["THREADS"]
    )
    config["val_loader"] = SpreadsheetDataLoader(
        val_files, config["vocab"]["object"], config["rows"], config["cols"], config["tokens"], threads=config["THREADS"]
    )
    config["test_loader"] = SpreadsheetDataLoader(
        test_files, config["vocab"]["object"], config["rows"], config["cols"], config["tokens"], threads=config["THREADS"]
    )

    ######## RETURN FINAL CONFIG ########
    return config


#################################################################################################
## BERT MODEL FUNCS

def display_bert_config(config):
    """
    Display the BERT configuration with non-serializable objects replaced by string placeholders.

    Parameters:
    - config: The configuration dictionary produced by setup_bert_config or similar.
    """
    # Create a deep copy of the config to avoid modifying the original in-place
    config_serializable = copy.deepcopy(config)

    # Convert the device object to string
    config_serializable["DEVICE"] = str(config_serializable["DEVICE"])

    # Replace tokenizer with a placeholder if present
    if "tokenizer" in config_serializable:
        config_serializable["tokenizer"] = "<ModernBert Tokenizer Object>"

    # Replace DataLoader objects with placeholders
    if "train_loader" in config_serializable:
        config_serializable["train_loader"] = "<Train BertLoader Object>"
    if "val_loader" in config_serializable:
        config_serializable["val_loader"] = "<Validation BertLoader Object>"
    if "test_loader" in config_serializable:
        config_serializable["test_loader"] = "<Test BertLoader Object>"

    # Print the resulting config in JSON form
    print("\nFinal BERT configuration:")
    print(json.dumps(config_serializable, indent=2))


def setup_bert_config(setup_config):
    """
    Sets up a configuration dictionary for a BERT-based workflow, including:
      - device selection
      - parallel threads
      - random seed
      - data directories for train/val/test
      - tokenizer creation (AutoTokenizer)
      - creation of three BertLoader objects

    Args:
        setup_config (dict): 
            A dictionary containing keys:
              {
                "device": "cuda:0" or "cpu" or "mps",
                "threads": int,
                "seed": int,
                "data_dir": str,       # base path like '../../data/farzan/'
                "data_ds": str,       # e.g. "manual" => 'manual_train', 'manual_val', 'manual_test'
                "bert_model": str,    # e.g. "answerdotai/ModernBERT-base"
                "rows": int,
                "cols": int,
                "tokens": int
              }

    Returns:
        dict: A config dictionary with relevant keys:
            {
              "bert_model": str,
              "DEVICE": torch.device,
              "THREADS": int,
              "data_dir": str,
              "data_ds": str,
              "train_dir": str,
              "val_dir": str,
              "test_dir": str,
              "rows": int,
              "cols": int,
              "tokens": int,
              "tokenizer": AutoTokenizer object,
              "train_loader": BertLoader object,
              "val_loader": BertLoader object,
              "test_loader": BertLoader object
            }
    """

    config = {}

    ######## BERT MODEL ########
    # We'll store the string name from your setup_config, e.g. "answerdotai/ModernBERT-base"
    config["model_name"] = setup_config["model_name"]

    ######## DATA DIR ########

    if os.path.isdir(setup_config["data_dir"]):
        config["data_dir"] = setup_config["data_dir"]
    else:
        raise ValueError(f"ERR: data_dir '{setup_config['data_dir']}' is not a valid path")

    ######## DEVICE ########

    if (
        setup_config["device"].startswith("cuda")
        and torch.cuda.is_available()
        and int(setup_config["device"].split(":")[1]) < torch.cuda.device_count()
    ):
        config["DEVICE"] = torch.device(setup_config["device"])
    elif (
        setup_config["device"].startswith("mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        config["DEVICE"] = torch.device("mps")
    else:
        config["DEVICE"] = torch.device("cpu")

    ######## THREADS ########

    remaining_threads = os.cpu_count() - (os.cpu_count() // setup_config["threads"])
    if remaining_threads >= 4:
        config["THREADS"] = setup_config["threads"]
    else:
        raise ValueError(f"ERR: {remaining_threads} threads remaining; need at least 4")

    ######## SEED ########

    set_seed(setup_config["seed"])

    ######## DATA_DS ########

    # e.g. "manual" => paths: data_dir/manual_train, data_dir/manual_val, data_dir/manual_test
    config["data_ds"] = setup_config["data_ds"]

    train_dir = os.path.join(config["data_dir"], f"{config['data_ds']}_train")
    val_dir   = os.path.join(config["data_dir"], f"{config['data_ds']}_val")
    test_dir  = os.path.join(config["data_dir"], f"{config['data_ds']}_test")

    missing_dirs = []
    for name, path in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        if not os.path.isdir(path):
            missing_dirs.append(name)

    if missing_dirs:
        raise ValueError(f"ERR: Missing dataset directories: {', '.join(missing_dirs)}")

    config["train_dir"] = train_dir
    config["val_dir"]   = val_dir
    config["test_dir"]  = test_dir

    ######## ROWS / COLS / TOKENS ########

    config["rows"]   = setup_config["rows"]
    config["cols"]   = setup_config["cols"]
    config["tokens"] = setup_config["tokens"]

    ######## TOKENIZER ########

    # Create the AutoTokenizer from your BERT model name
    config["tokenizer"] = AutoTokenizer.from_pretrained(config["model_name"])

    ######## DATALOADERS ########

    # Gather file lists for each split
    train_files, _ = get_fileList(config["train_dir"])
    val_files,   _ = get_fileList(config["val_dir"])
    test_files,  _ = get_fileList(config["test_dir"])

    # Create three BertLoader instances
    config["train_loader"] = BertLoader(
        file_paths=train_files,
        tokenizer=config["tokenizer"],
        max_rows=config["rows"],
        max_cols=config["cols"],
        pad_length=config["tokens"],
        threads=config["THREADS"]
    )
    config["val_loader"] = BertLoader(
        file_paths=val_files,
        tokenizer=config["tokenizer"],
        max_rows=config["rows"],
        max_cols=config["cols"],
        pad_length=config["tokens"],
        threads=config["THREADS"]
    )
    config["test_loader"] = BertLoader(
        file_paths=test_files,
        tokenizer=config["tokenizer"],
        max_rows=config["rows"],
        max_cols=config["cols"],
        pad_length=config["tokens"],
        threads=config["THREADS"]
    )

    return config