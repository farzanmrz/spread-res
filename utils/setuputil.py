# General imports
import copy
import importlib
import json
import os
import time

import torch
from transformers import AutoTokenizer

from classes import Loader

# Reload the selfutil module and import required functions
from utils import selfutil

importlib.reload(selfutil)
importlib.reload(Loader)

from classes.Loader import LoaderBert, LoaderSimple

# Import utils and classes needed
from utils.selfutil import create_embeddings, get_fileList, get_vocab, set_seed


# Define a function to validate the input configuration dictionary
def h_valInput(input_config):

    # Build a list of keys from input_config that are not in the allowed keys set
    invalid_keys = [
        key
        for key in input_config
        if key
        not in {
            "env",
            "approach",
            "DEVICE",
            "THREADS",
            "seed",
            "model_base",
            "model_name",
            "rows",
            "cols",
            "tokens",
            "data_ds",
            "data_dir",
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "hidden_act",
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "max_position_embeddings",
            "type_vocab_size",
            "initializer_range",
            "layer_norm_eps",
            "pad_token_id",
            "gradient_checkpointing",
            "batch_size",
            "lr",
            "mu",
            "epochs",
            "patience",
            "save_int",
            "save_dir",
        }
    ]

    # If any invalid keys are found, raise a ValueError with the list of invalid keys
    if invalid_keys:
        raise KeyError(f"Invalid Input Keys {invalid_keys}")


def h_env(input_config):
    """Helper function to validate environment and approach."""

    # Define empty config dict, valid environments/approaches
    config = {}
    valid_envs = ["gcp", "bvm", "local", "colab"]
    valid_approaches = ["simple", "saffu", "bert", "rnn"]

    # Check if env/approach provided is valid
    if input_config["env"] not in valid_envs:
        raise ValueError(f"ERR: env must be one of {valid_envs}")
    if input_config["approach"] not in valid_approaches:
        raise ValueError(f"ERR: approach must be one of {valid_approaches}")

    # Update config with validated environment and approach
    config.update({"env": input_config["env"], "approach": input_config["approach"]})

    # Return config
    return config


def h_device(input_config, config):
    """Helper function to validate and setup device configuration."""

    # Try catch block to default to cpu for any error
    try:

        # Check if device is provided in input_config
        if "DEVICE" not in input_config:

            # Local = MPS
            if (
                config["env"] == "local"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                config["DEVICE"] = torch.device("mps:0")

            # Colab/GCP/BVM = CUDA if available
            elif config["env"] in ["colab", "gcp", "bvm"] and torch.cuda.is_available():
                config["DEVICE"] = torch.device("cuda:0")

            # Default to CPU
            else:
                print("\nGPU not available, defaulting to CPU\n")
                config["DEVICE"] = torch.device("cpu")

        # Else if device provided
        else:
            #         # Original device validation logic for when device is specified
            device_config = input_config["DEVICE"]

            # # Validate device string contains index specification
            # if ":" not in device_config:
            #     raise ValueError("ERR: Specify device index (e.g., cuda:0, mps:0)")

            # Check for CUDA device in non-local environments
            if (
                config["env"] != "local"
                and device_config.startswith("cuda")
                and torch.cuda.is_available()
                and int(device_config.split(":")[1]) < torch.cuda.device_count()
            ):
                config["DEVICE"] = torch.device(device_config)

            # Check for MPS device in local environment
            elif (
                config["env"] == "local"
                and device_config.startswith("mps")
                and device_config.split(":")[1] == "0"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                config["DEVICE"] = torch.device(device_config)
            else:
                print(
                    "\nSpecified GPU not available or CPU specified as DEVICE, therefore device is defaulting to CPU\n"
                )
                config["DEVICE"] = torch.device("cpu")

    # If any error then default to CPU
    except Exception as e:
        print(f"ERR: {e} \nDefaulting to CPU")
        config["DEVICE"] = torch.device("cpu")

    return config


def h_threads(input_config, config):
    """Helper function to validate and setup thread configuration."""

    # If threads param given then
    if "threads" in input_config:

        # Validate threads parameter is numeric
        if not isinstance(input_config["threads"], (int, float)):
            raise ValueError("ERR: threads must be a number")

        # If bvm no more than 20 allowed
        if config["env"] == "bvm" and int(input_config["threads"]) > 20:
            raise ValueError("ERR: BVM environment cannot request more than 20 threads")

        # Always leave 2 threads at end for other worker stuff
        if os.cpu_count() - int(input_config["threads"]) < 2:
            raise ValueError(
                f"ERR: Must leave at least 2 threads free (requested {input_config['threads']})"
            )

        # Set threads in config dict finally as int and max between 1
        config["THREADS"] = max(1, int(input_config["threads"]))

    # ELse if threads param dont exist then set threads based on env
    else:
        config["THREADS"] = (
            16
            if config["env"] == "bvm" and os.cpu_count() >= 18
            else max(1, os.cpu_count() - 2)
        )

    return config


def h_seed(input_config, config):
    """Helper function to set seed configuration."""
    # Set torch printing options
    torch.set_printoptions(precision=4, sci_mode=False)

    # Disable efficient sdp globally
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Set seed value in config default unless other provided
    config["seed"] = input_config.get("seed", 0)

    # Apply seed setting
    set_seed(config["seed"])

    # Return updated configuration
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
        config["model_base"] = "saffu"  # Force saffu
    elif config["approach"] == "bert":
        # Use provided model_base or default to bert-tiny
        config["model_base"] = input_config.get("model_base", "prajjwal1/bert-tiny")

    # Set model_name as provided
    config["model_name"] = input_config["model_name"]

    ######## CONTEXT PARAMS ########

    # Updte context with default values unless other provided
    config.update(
        {
            "rows": input_config.get("rows", 100),
            "cols": input_config.get("cols", 100),
            "tokens": input_config.get("tokens", 32),
        }
    )

    return config


def h_data(config, input_config):
    """Helper function to setup data-related configurations."""

    ######## DATA DIR & DATASET ########
    # Set default data directory if not provided
    data_dir = input_config.get("data_dir", "../data")

    # Validate data directory and update config
    if not os.path.isdir(data_dir):
        raise ValueError(f"ERR: data_dir '{data_dir}' is not a valid path")
    config.update({"data_ds": input_config["data_ds"], "data_dir": data_dir})

    ######## DATA DIRECTORIES ########
    # Create directory paths
    train_dir = os.path.join(config["data_dir"], f"{input_config['data_ds']}_train")
    val_dir = os.path.join(config["data_dir"], f"{input_config['data_ds']}_val")
    test_dir = os.path.join(config["data_dir"], f"{input_config['data_ds']}_test")

    # Validate directories exist
    missing_dirs = [
        dir_name
        for dir_name, path in {
            "train": train_dir,
            "val": val_dir,
            "test": test_dir,
        }.items()
        if not os.path.isdir(path)
    ]
    if missing_dirs:
        raise ValueError(f"ERR: Missing dataset directories: {', '.join(missing_dirs)}")

    # Update config after validation
    config.update({"train_dir": train_dir, "val_dir": val_dir, "test_dir": test_dir})

    return config


def h_vocab(config, input_config):
    """Helper function to setup vocabulary only for simple/rnn approaches."""
    if config["approach"] not in ["simple", "rnn"]:
        return config

    ######## VOCAB ########
    # Validate vocab parameters
    if (
        not isinstance(input_config["vocab_size"], int)
        or not 4 <= input_config["vocab_size"] <= 2000000
    ):
        raise ValueError(
            f"ERR: vocab_size '{input_config['vocab_size']}' must be an integer between 4 and 2,000,000"
        )

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
        threads=config["THREADS"],
    )

    ######## WVS ########
    config["wvs"] = create_embeddings(config["vocab"])
    config.update(
        {
            "vocab_size": config["wvs"].shape[0],
            "vocab_space": vocab_space,
            "vocab_case": vocab_case,
        }
    )

    return config


def h_rnn(config, input_config):
    """Helper function to setup RNN-specific parameters."""
    if config["approach"] != "rnn":
        return config

    ######## RNN PARAMETERS ########
    config.update(
        {
            "hidden_size": input_config.get("hidden_size", 128),
            "num_hidden_layers": input_config.get("num_hidden_layers", 2),
            "hidden_dropout_prob": input_config.get("hidden_dropout_prob", 0.05),
            "hidden_act": input_config.get("hidden_act", "relu"),
        }
    )

    return config


def h_name(config):
    """Helper function to generate the save name based on the configuration."""

    ### BASE FORMAT SAME FOR ALL APPROACHES ###

    # 1. env 1st letter + model name + ds + current time
    return (
        config["env"].lower()[:1]
        + config["model_name"]
        + "_"
        + config["data_ds"]
        + "_"
        + time.strftime("%m%d%H%M")
    )


def h_training(config, input_config):
    """Helper function to setup training parameters and call h_name for save name generation."""
    ######## TRAINING PARAMS ########
    # Define default save dir if exists else provided one check
    save_dir = input_config.get("save_dir", "../models/")
    if not os.path.isdir(save_dir):
        raise ValueError(f"ERR: save_dir '{save_dir}' is not a valid path")

    # Update config
    config.update(
        {
            "batch_size": input_config["batch_size"],
            "lr": input_config["lr"],
            "mu": input_config["mu"],
            "epochs": input_config["epochs"],
            "patience": input_config["patience"],
            "save_int": input_config["save_int"],
            "save_dir": save_dir,
        }
    )

    ######## SAVE NAME ########
    config["save_name"] = h_name(config)

    return config


def h_simpleloader(config):
    """Helper function to setup LoaderSimple for vocabulary-based approaches."""
    # Generate file lists
    train_files, _ = get_fileList(config["train_dir"])
    val_files, _ = get_fileList(config["val_dir"])
    test_files, _ = get_fileList(config["test_dir"])

    # Create Loaders
    config.update(
        {
            "train_loader": LoaderSimple(
                file_paths=train_files,
                vocab=config["vocab"],
                max_rows=config["rows"],
                max_cols=config["cols"],
                pad_length=config["tokens"],
                threads=config["THREADS"],
            ),
            "val_loader": LoaderSimple(
                file_paths=val_files,
                vocab=config["vocab"],
                max_rows=config["rows"],
                max_cols=config["cols"],
                pad_length=config["tokens"],
                threads=config["THREADS"],
            ),
            "test_loader": LoaderSimple(
                file_paths=test_files,
                vocab=config["vocab"],
                max_rows=config["rows"],
                max_cols=config["cols"],
                pad_length=config["tokens"],
                threads=config["THREADS"],
            ),
        }
    )
    return config


def h_setupbert(config, input_config):
    """Helper function to setup BERT configuration, tokenizer, and loaders."""

    # First set up tokenizer as before
    config["tokenizer"] = AutoTokenizer.from_pretrained(config["model_base"])

    # Setup BERT configuration with defaults from documentation
    bert_config = {
        "vocab_size": 30522,  # Default from docs
        "hidden_size": input_config.get("hidden_size", 128),
        "num_hidden_layers": input_config.get("num_hidden_layers", 1),
        "num_attention_heads": input_config.get("num_attention_heads", 4),
        "intermediate_size": input_config.get("intermediate_size", 512),
        "hidden_act": input_config.get("hidden_act", "gelu"),
        "hidden_dropout_prob": input_config.get("hidden_dropout_prob", 0.1),
        "attention_probs_dropout_prob": input_config.get(
            "attention_probs_dropout_prob", 0.1
        ),
        "max_position_embeddings": input_config.get("max_position_embeddings", 64),
        "type_vocab_size": input_config.get("type_vocab_size", 2),
        "initializer_range": input_config.get("initializer_range", 0.02),
        "layer_norm_eps": input_config.get("layer_norm_eps", 1e-12),
        "pad_token_id": input_config.get("pad_token_id", 0),
        "gradient_checkpointing": input_config.get("gradient_checkpointing", False),
    }

    # Update config with BERT parameters
    config.update(bert_config)

    # Setup loaders as before
    train_files, _ = get_fileList(config["train_dir"])
    val_files, _ = get_fileList(config["val_dir"])
    test_files, _ = get_fileList(config["test_dir"])

    config.update(
        {
            "train_loader": LoaderBert(
                file_paths=train_files,
                tokenizer=config["tokenizer"],
                max_rows=config["rows"],
                max_cols=config["cols"],
                pad_length=config["tokens"],
                threads=config["THREADS"],
            ),
            "val_loader": LoaderBert(
                file_paths=val_files,
                tokenizer=config["tokenizer"],
                max_rows=config["rows"],
                max_cols=config["cols"],
                pad_length=config["tokens"],
                threads=config["THREADS"],
            ),
            "test_loader": LoaderBert(
                file_paths=test_files,
                tokenizer=config["tokenizer"],
                max_rows=config["rows"],
                max_cols=config["cols"],
                pad_length=config["tokens"],
                threads=config["THREADS"],
            ),
        }
    )

    return config


def setup_config(input_config):
    """Sets up the configuration for model training with modular helper functions."""

    # 1. Validate input configuration keys
    h_valInput(input_config)

    # 2. Set environment
    config = h_env(input_config)

    # 3. Setup device
    config = h_device(input_config, config)

    # 4. Setup threads
    config = h_threads(input_config, config)

    # 5. Setup seed
    config = h_seed(input_config, config)

    # 6. Setup model
    config = h_model(config, input_config)

    # 7. Setup data
    config = h_data(config, input_config)

    # 8. For simple/rnn model do the following
    if config["approach"] in ["simple", "rnn"]:

        # 8.1. Custom vocab object setup
        config = h_vocab(config, input_config)

        # 8.2. Custom dataset class made with vocab
        config = h_simpleloader(config)

        # 8.3. Specifically RNN do further setup
        if config["approach"] == "rnn":
            config = h_rnn(config, input_config)

    # 9. For BERT model setup its specific configurations
    elif config["approach"] == "bert":
        config = h_setupbert(config, input_config)

    # 10. For SAFFU model setup its specific configurations
    elif config["approach"] == "saffu":
        pass

    # 11. Training parameters and save name
    config = h_training(config, input_config)

    # 12. Display the config
    h_displayConfig(config)

    return config


def h_displayConfig(config):
    """Display configuration in a readable format."""
    serializable_config = {
        k: (
            v
            if isinstance(v, (str, int, float, bool, type(None)))
            else f"<{v.__class__.__name__}>"
        )
        for k, v in config.items()
    }

    # Print the json as per requirement
    print(f"\nFINAL CONFIG:")
    print(json.dumps(serializable_config, indent=2))

    # Print base info needed
    print(f'\nDEVICE: {config["DEVICE"]} | THREADS: {config["THREADS"]}')
