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

# Simple setup display func
def display_simple_config(config):
   """
   Display the configuration with non-serializable objects replaced by string placeholders.
   Parameters:
   - config: The configuration dictionary.
   """
   # Create a deep copy of the config to avoid modifying the original  
   config_serializable = copy.deepcopy(config)

   # Replace non-serializable objects with string representations
   config_serializable["DEVICE"] = str(config_serializable["DEVICE"])
   config_serializable["vocab"] = "<Vocab Object>" 
   config_serializable["wvs"] = "<Embedding Tensor>"
   config_serializable["train_loader"] = "<Train SpreadsheetDataLoader Object>"
   config_serializable["val_loader"] = "<Validation SpreadsheetDataLoader Object>"
   config_serializable["test_loader"] = "<Test SpreadsheetDataLoader Object>"

   # Create ordered dictionary for better display
   ordered_config = {
       # Environment and Model Info
       "env": config_serializable["env"],
       "approach": config_serializable["approach"],
       "model_name": config_serializable["model_name"],

       # System Configuration
       "DEVICE": config_serializable["DEVICE"],
       "THREADS": config_serializable["THREADS"], 
       "seed": config_serializable["seed"],

       # Data Configuration
       "data_dir": config_serializable["data_dir"],
       "data_ds": config_serializable["data_ds"],

       # Data Directories  
       "train_dir": config_serializable["train_dir"],
       "val_dir": config_serializable["val_dir"],
       "test_dir": config_serializable["test_dir"],

       # Model Parameters
       "rows": config_serializable["rows"],
       "cols": config_serializable["cols"],
       "tokens": config_serializable["tokens"],

       # Vocabulary Configuration
       "vocab_size": config_serializable["vocab_size"],
       "vocab_space": config_serializable["vocab_space"], 
       "vocab_case": config_serializable["vocab_case"],
       "vocab": config_serializable["vocab"],
       "wvs": config_serializable["wvs"],

       # Data Loaders
       "train_loader": config_serializable["train_loader"],
       "val_loader": config_serializable["val_loader"],
       "test_loader": config_serializable["test_loader"],

       # Training Configuration
       "batch": config_serializable["batch"],
       "lr": config_serializable["lr"],
       "mu": config_serializable["mu"],
       "epochs": config_serializable["epochs"], 
       "patience": config_serializable["patience"],
       "save_int": config_serializable["save_int"],
       "save_dir": config_serializable["save_dir"],
       "save_name": config_serializable["save_name"]
   }

   # Pretty-print the configuration  
   print("\nFinal configuration:")
   print(json.dumps(ordered_config, indent=2))
    
    
# Define the simple_setup function
def setup_simple_config(setup_config):
    
    # Initialize the config dictionary
    config = {}
    
    ######## ENVIRONMENT & MODEL INFO ########
    
    # Validate and set environment
    valid_envs = ["gcp", "bvm", "local", "colab"]
    if setup_config["env"] not in valid_envs:
        raise ValueError(f"ERR: env must be one of {valid_envs}")
    config["env"] = setup_config["env"]
    
    # Validate and set approach
    valid_approaches = ["simple", "saffu", "bert"]
    if setup_config["approach"] not in valid_approaches:
        raise ValueError(f"ERR: approach must be one of {valid_approaches}")
    config["approach"] = setup_config["approach"]
    
    # Set model name
    config["model_name"] = setup_config["model_name"]

    ######## DATA DIR & DATASET ########
    
    # Set data directory and dataset name
    if os.path.isdir(setup_config["data_dir"]):
        config["data_dir"] = setup_config["data_dir"]
        config["data_ds"] = setup_config["data_ds"]
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
    config["seed"] = setup_config["seed"] 

    ######## THREADS ########

    # # Validate and set THREADS
    # remaining_threads = os.cpu_count() - (os.cpu_count() // setup_config["threads"])
    # if remaining_threads >= 4:
    #     config["THREADS"] = setup_config["threads"]
    # else:
    #     raise ValueError(f"ERR: {remaining_threads} threads remaining leave at least 4")
    # Validate thread input type
    if not isinstance(setup_config["threads"], (int, float)):
        raise ValueError("ERR: threads must be a number")
    
    # Convert to integer and validate minimum value
    requested_threads = int(setup_config["threads"])  # Convert float to int
    total_threads = os.cpu_count()
    remaining_threads = total_threads - requested_threads
    
    # Check if we're leaving enough threads free
    if remaining_threads < 4:
        raise ValueError(f"ERR: Using {requested_threads} threads would only leave {remaining_threads} threads free. Must leave at least 4 threads.")
    
    # Set threads to max of requested (as int) or 1
    config["THREADS"] = max(1, requested_threads)

    ######## SEED ########

    # Set the random seed
    set_seed(setup_config["seed"])

    ######## VOCAB ########

    # Validate vocab parameters
    if not isinstance(setup_config["vocab_size"], int) or not 4 <= setup_config["vocab_size"] <= 2000000:
        raise ValueError(f"ERR: vocab_size '{setup_config['vocab_size']}' must be an integer between 4 and 2,000,000")
    
    vocab_space = setup_config.get("vocab_space", True)
    if not isinstance(vocab_space, bool):
        vocab_space = True
        
    vocab_case = setup_config.get("vocab_case", "lower")
    if vocab_case not in ["both", "upper", "lower"]:
        vocab_case = "lower"
    
    # Generate vocab object using train_dir as vocab_dir
    train_dir = os.path.join(config["data_dir"], f"{setup_config['data_ds']}_train")
    config["vocab"] = get_vocab(
        train_dir,
        setup_config["vocab_size"],
        space=vocab_space,
        case=vocab_case,
        threads=config["THREADS"]
    )

    ######## WVS ########

    # Generate the embedding matrix (wvs) using create_embeddings
    config["wvs"] = create_embeddings(config["vocab"])
    
    # Store the actual vocab size and other parameters
    config["vocab_size"] = config["wvs"].shape[0]  
    config["vocab_space"] = vocab_space
    config["vocab_case"] = vocab_case

    ######## DATA_DS ########

    # Validate and set train, val, test directories
    config["train_dir"] = train_dir
    config["val_dir"] = os.path.join(config["data_dir"], f"{setup_config['data_ds']}_val")
    config["test_dir"] = os.path.join(config["data_dir"], f"{setup_config['data_ds']}_test")

    missing_dirs = [dir_name for dir_name, path in 
        {"train": config["train_dir"], "val": config["val_dir"], "test": config["test_dir"]}.items() 
        if not os.path.isdir(path)]

    if missing_dirs:
        raise ValueError(f"ERR: Missing dataset directories: {', '.join(missing_dirs)}")

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
        train_files, config["vocab"], config["rows"], config["cols"], config["tokens"], threads=config["THREADS"]
    )
    config["val_loader"] = SpreadsheetDataLoader(
        val_files, config["vocab"], config["rows"], config["cols"], config["tokens"], threads=config["THREADS"]
    )
    config["test_loader"] = SpreadsheetDataLoader(
        test_files, config["vocab"], config["rows"], config["cols"], config["tokens"], threads=config["THREADS"]
    )
    
    
    ######## TRAINING ########
    
    # Validate and set training parameters
    config["batch"] = setup_config["batch"]
    config["lr"] = setup_config["lr"]
    config["mu"] = setup_config["mu"]
    config["epochs"] = setup_config["epochs"]
    config["patience"] = setup_config["patience"]
    config["save_int"] = setup_config["save_int"]
    config["save_dir"] = setup_config["save_dir"]

    ######## SAVE NAME ########
    
    # Create vocab config string for name
    case_prefix = {"both": "b", "upper": "u", "lower": "l"}[config["vocab_case"]]
    space_str = "Sp" if config["vocab_space"] else "Nsp"
    vocab_str = f"{case_prefix}{space_str}{config['vocab_size']//1000}k"
    
    # Generate model name
    save_name = "__".join([
        "_".join([config['env'], config['approach'], config['model_name'], f"s{config['seed']}"]),  # Added seed here
        "_".join([config['data_ds'], f"{config['rows']}x{config['cols']}x{config['tokens']}"]),
        vocab_str,
        f"b{config['batch']}" + \
        f"lr{config['lr']:.0e}".replace('e-0', 'e-') + \
        f"e{config['epochs']}" + \
        f"p{config['patience']}"
    ])
    config["save_name"] = save_name

    
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