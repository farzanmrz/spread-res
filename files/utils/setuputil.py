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

def h_env(input_config):
    """Helper function to validate and setup environment-related configurations."""
    config = {}
    
    ######## ENVIRONMENT ########
    valid_envs = ["gcp", "bvm", "local", "colab"]
    valid_approaches = ["simple", "saffu", "bert", "rnn"]
    
    if input_config["env"] not in valid_envs:
        raise ValueError(f"ERR: env must be one of {valid_envs}")
    if input_config["approach"] not in valid_approaches:
        raise ValueError(f"ERR: approach must be one of {valid_approaches}")
        
    config.update({
        "env": input_config["env"],
        "approach": input_config["approach"]
    })
    
    ######## DEVICE ########
    device_config = input_config["device"]
    if (device_config.startswith("cuda") and torch.cuda.is_available() 
        and int(device_config.split(":")[1]) < torch.cuda.device_count()):
        config["DEVICE"] = torch.device(device_config)
    elif device_config.startswith("mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        config["DEVICE"] = torch.device("mps")
    else:
        config["DEVICE"] = torch.device("cpu")
    
    ######## THREADS ########
    if not isinstance(input_config["threads"], (int, float)):
        raise ValueError("ERR: threads must be a number")
    
    threads = int(input_config["threads"])
    if (os.cpu_count() - threads) < 4:
        raise ValueError(f"ERR: Must leave at least 4 threads free (requested {threads})")
    config["THREADS"] = max(1, threads)
    
    ######## SEED ########
    config["seed"] = input_config["seed"]
    set_seed(config["seed"])
    
    return config

def h_model(config, input_config):
    """Helper function to setup model-related configurations."""
    
    ######## MODEL ########
    # Ensure model_name is always provided
    if "model_name" not in input_config:
        raise ValueError("ERR: model_name must be provided for all approaches")
    
    # Set model_base based on approach, overriding any provided value if needed
    if config["approach"] in ["simple", "rnn"]:
        config["model_base"] = "glove50"  # Force glove50 for simple/rnn
    elif config["approach"] == "saffu":
        config["model_base"] = "saffu"    # Force saffu
    elif config["approach"] == "bert":
        # Use provided model_base or default to bert-tiny
        config["model_base"] = input_config.get("model_base", "prajjwal1/bert-tiny")
    
    # Set model_name as provided
    config["model_name"] = input_config["model_name"]
    
    ######## CONTEXT PARAMS ########
    config.update({
        "rows": input_config["rows"],
        "cols": input_config["cols"],
        "tokens": input_config["tokens"]
    })
    
    return config

def h_data(config, input_config):
    """Helper function to setup data-related configurations."""
    ######## DATA DIR & DATASET ########
    if not os.path.isdir(input_config["data_dir"]):
        raise ValueError(f"ERR: data_dir '{input_config['data_dir']}' is not a valid path")
    
    config.update({
        "data_ds": input_config["data_ds"],
        "data_dir": input_config["data_dir"]
    })
    
    ######## DATA DIRECTORIES ########
    # Create directory paths
    train_dir = os.path.join(config["data_dir"], f"{input_config['data_ds']}_train")
    val_dir = os.path.join(config["data_dir"], f"{input_config['data_ds']}_val")
    test_dir = os.path.join(config["data_dir"], f"{input_config['data_ds']}_test")
    
    # Validate directories exist
    missing_dirs = [
        dir_name for dir_name, path in 
        {"train": train_dir, "val": val_dir, "test": test_dir}.items() 
        if not os.path.isdir(path)
    ]
    if missing_dirs:
        raise ValueError(f"ERR: Missing dataset directories: {', '.join(missing_dirs)}")
    
    # Update config after validation
    config.update({
        "train_dir": train_dir,
        "val_dir": val_dir,
        "test_dir": test_dir
    })
    
    return config

def h_vocab(config, input_config):
    """Helper function to setup vocabulary only for simple/rnn approaches."""
    if config["approach"] not in ["simple", "rnn"]:
        return config
        
    ######## VOCAB ########
    # Validate vocab parameters
    if not isinstance(input_config["vocab_size"], int) or not 4 <= input_config["vocab_size"] <= 2000000:
        raise ValueError(f"ERR: vocab_size '{input_config['vocab_size']}' must be an integer between 4 and 2,000,000")
    
    vocab_space = input_config.get("vocab_space", True)
    if not isinstance(vocab_space, bool):
        vocab_space = True
        
    vocab_case = input_config.get("vocab_case", "lower")
    if vocab_case not in ["both", "upper", "lower"]:
        vocab_case = "lower"
    
    # Generate vocab object using train_dir
    config["vocab"] = get_vocab(
        config["train_dir"],
        input_config["vocab_size"],
        space=vocab_space,
        case=vocab_case,
        threads=config["THREADS"]
    )
    
    ######## WVS ########
    config["wvs"] = create_embeddings(config["vocab"])
    config.update({
        "vocab_size": config["wvs"].shape[0],
        "vocab_space": vocab_space,
        "vocab_case": vocab_case
    })
    
    return config

def h_rnn(config, setup_config):
    """Helper function to setup RNN-specific parameters."""
    if config["approach"] != "rnn":
        return config
        
    ######## RNN PARAMETERS ########
    config.update({
        "hidden_dim": setup_config.get("hidden_dim", 128),
        "rnn_layers": setup_config.get("rnn_layers", 2),
        "dropout_rate": setup_config.get("dropout_rate", 0.05),
        "nonlinearity": setup_config.get("nonlinearity", "relu")
    })
    
    return config

def h_training(config, setup_config):
    """Helper function to setup training parameters and generate save name."""
    ######## TRAINING PARAMS ########
    config.update({
        "batch": setup_config["batch"],
        "lr": setup_config["lr"],
        "mu": setup_config["mu"],
        "epochs": setup_config["epochs"],
        "patience": setup_config["patience"],
        "save_int": setup_config["save_int"],
        "save_dir": setup_config["save_dir"]
    })

    ######## SAVE NAME ########
    # Basic components (common across all approaches)
    env_map = {"gcp": "g", "local": "l", "bvm": "b", "colab": "c"}
    env_abbr = env_map[config["env"]]
    app_prefix = config["approach"][:3]
    base_name = f"{env_abbr}{app_prefix}{config['seed']}"
    
    # Context dims component
    dims = f"{config['model_name']}_{config['data_ds']}_{config['rows']}x{config['cols']}x{config['tokens']}"
    
    # Training params component
    train_params = (f"bsz{config['batch']}lr{config['lr']:.0e}"
                   .replace('e-0', 'e-') + 
                   f"ep{config['epochs']}pa{config['patience']}")

    # Approach-specific components
    if config["approach"] in ["simple", "rnn"]:
        # Vocab string for simple/rnn approaches
        case_prefix = {"both": "b", "upper": "u", "lower": "l"}[config["vocab_case"]]
        space_str = "Sp" if config["vocab_space"] else "Nsp"
        vocab_str = f"{case_prefix}{space_str}{config['vocab_size']//1000}k"
        
        # Additional RNN-specific component
        if config["approach"] == "rnn":
            rnn_str = f"_rnn{config['rnn_layers']}hid{config['hidden_dim']}"
        else:
            rnn_str = ""
            
        save_name = f"{base_name}_{dims}_{vocab_str}_{train_params}{rnn_str}"
    
    elif config["approach"] == "bert":
        # BERT models don't need vocab string
        save_name = f"{base_name}_{dims}_{train_params}"
    
    elif config["approach"] == "saffu":
        # SAFFU models don't need vocab string
        save_name = f"{base_name}_{dims}_{train_params}"
    
    config["save_name"] = save_name
    return config

def h_simpleloader(config):
    """Helper function to setup SpreadsheetDataLoaders for simple/rnn approaches."""
    ######## SIMPLE LOADERS ########
    # Generate file lists
    train_files, _ = get_fileList(config["train_dir"])
    val_files, _ = get_fileList(config["val_dir"])
    test_files, _ = get_fileList(config["test_dir"])

    # Create SpreadsheetDataLoaders
    config.update({
        "train_loader": SpreadsheetDataLoader(
            train_files, config["vocab"], 
            config["rows"], config["cols"], config["tokens"], 
            threads=config["THREADS"]
        ),
        "val_loader": SpreadsheetDataLoader(
            val_files, config["vocab"], 
            config["rows"], config["cols"], config["tokens"], 
            threads=config["THREADS"]
        ),
        "test_loader": SpreadsheetDataLoader(
            test_files, config["vocab"], 
            config["rows"], config["cols"], config["tokens"], 
            threads=config["THREADS"]
        )
    })
    
    return config

def h_bertloader(config):
    """Helper function to setup BertLoader for BERT approach."""
    ######## BERT LOADERS ########
    # Generate file lists
    train_files, _ = get_fileList(config["train_dir"])
    val_files, _ = get_fileList(config["val_dir"])
    test_files, _ = get_fileList(config["test_dir"])

    # Create BertLoader objects
    config.update({
        "train_loader": BertLoader(
            train_files, config["tokenizer"], 
            config["rows"], config["cols"], config["tokens"], 
            threads=config["THREADS"]
        ),
        "val_loader": BertLoader(
            val_files, config["tokenizer"], 
            config["rows"], config["cols"], config["tokens"], 
            threads=config["THREADS"]
        ),
        "test_loader": BertLoader(
            test_files, config["tokenizer"], 
            config["rows"], config["cols"], config["tokens"], 
            threads=config["THREADS"]
        )
    })
    
    return config



def h_tokbert(config):
   """Helper function to setup BERT tokenizer using model_base."""
   ######## TOKENIZER ########
   config["tokenizer"] = AutoTokenizer.from_pretrained(config["model_base"])
   return config

def setup_config(setup_config):
    """Sets up the configuration for model training with modular helper functions."""
    ######## ENVIRONMENT ########
    config = h_env(setup_config)

    ######## MODEL ########
    config = h_model(config, setup_config)

    ######## DATA ########
    config = h_data(config, setup_config)

    ######## APPROACH-SPECIFIC SETUP ########
    if config["approach"] in ["simple", "rnn"]:
       
        ######## VOCAB ########
        config = h_vocab(config, setup_config)

        ######## SIMPLE LOADERS ########
        config = h_simpleloader(config)

        ######## RNN PARAMS ########
        if config["approach"] == "rnn":
            config = h_rnn(config, setup_config)

    ######## BERT-SPECIFIC ########
    elif config["approach"] == "bert":
        
        ######## TOKENIZER ########
        config = h_tokbert(config)
        
        ######## BERT LOADERS ########
        config = h_bertloader(config)

    ######## SAFFU-SPECIFIC ########
    elif config["approach"] == "saffu":
       pass

    ######## TRAINING & SAVE NAME ########
    config = h_training(config, setup_config)

    return config

def display_config(config):
    """Display the current configuration settings."""
    config_serializable = copy.deepcopy(config)
    config_serializable["DEVICE"] = str(config_serializable["DEVICE"])
    
    # Base configuration that exists for all approaches
    ordered_config = {
        # Environment Info
        "env": config_serializable["env"],
        "approach": config_serializable["approach"],
        
        # Model Info
        "model_base": config_serializable["model_base"],
        "model_name": config_serializable["model_name"],
        
        # Context Parameters
        "rows": config_serializable["rows"],
        "cols": config_serializable["cols"],
        "tokens": config_serializable["tokens"],
        
        # System Configuration
        "DEVICE": config_serializable["DEVICE"],
        "THREADS": config_serializable["THREADS"],
        "seed": config_serializable["seed"],
        
        # Data Configuration
        "data_ds": config_serializable["data_ds"],
        "data_dir": config_serializable["data_dir"],
        "train_dir": config_serializable["train_dir"],
        "val_dir": config_serializable["val_dir"],
        "test_dir": config_serializable["test_dir"],
    }

    # Add vocabulary configuration if approach is simple/rnn
    if config_serializable["approach"] in ["simple", "rnn"]:
        vocab_config = {
            # Vocabulary Configuration
            "vocab_size": config_serializable["vocab_size"],
            "vocab_space": config_serializable["vocab_space"],
            "vocab_case": config_serializable["vocab_case"],
            "vocab": "<Vocab Object>",
            "wvs": "<Embedding Matrix>",
            
            # DataLoader Configuration
            "train_loader": "<SpreadsheetDataLoader Object>",
            "val_loader": "<SpreadsheetDataLoader Object>",
            "test_loader": "<SpreadsheetDataLoader Object>"
        }
        ordered_config.update(vocab_config)
        
    # Add BERT-specific configuration if approach is bert
    elif config_serializable["approach"] == "bert":
        bert_config = {
            # BERT Configuration
            "tokenizer": "<BERT Tokenizer Object>",
            
            # DataLoader Configuration
            "train_loader": "<BertLoader Object>",
            "val_loader": "<BertLoader Object>",
            "test_loader": "<BertLoader Object>"
        }
        ordered_config.update(bert_config)
        
    # Add training configuration for all approaches
    ordered_config.update({
        # Training Configuration
        "batch": config_serializable["batch"],
        "lr": config_serializable["lr"],
        "mu": config_serializable["mu"],
        "epochs": config_serializable["epochs"],
        "patience": config_serializable["patience"],
        "save_int": config_serializable["save_int"],
        "save_dir": config_serializable["save_dir"],
        "save_name": config_serializable["save_name"]
    })

    # Add RNN-specific configuration if applicable
    if config_serializable["approach"] == "rnn":
        rnn_config = {
            # RNN Parameters
            "hidden_dim": config_serializable["hidden_dim"],
            "rnn_layers": config_serializable["rnn_layers"],
            "dropout_rate": config_serializable["dropout_rate"],
            "nonlinearity": config_serializable["nonlinearity"]
        }
        ordered_config.update(rnn_config)

    print(f"\nConfiguration for {config_serializable['approach'].upper()} approach:")
    print(json.dumps(ordered_config, indent=2))


