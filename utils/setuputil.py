# General imports
import copy
import importlib
import json
import os

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
    # Extract device configuration string
    device_config = input_config["device"]

    # Validate device string contains index specification
    if ":" not in device_config:
        raise ValueError("ERR: Specify device index (e.g., cuda:0, mps:0)")

    # Check for CUDA device in non-local environments
    if (
        config["env"] != "local"
        and device_config.startswith("cuda")
        and torch.cuda.is_available()
        and int(device_config.split(":")[1]) < torch.cuda.device_count()
    ):
        # Set device to specified CUDA device
        config["DEVICE"] = torch.device(device_config)

    # Check for MPS device in local environment
    elif (
        config["env"] == "local"
        and device_config.startswith("mps")
        and device_config.split(":")[1] == "0"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        # Set device to MPS
        config["DEVICE"] = torch.device(device_config)

    # Default to CPU if neither CUDA nor MPS is available
    else:
        print("\nGPU DNE defaulting to CPU\n")
        config["DEVICE"] = torch.device("cpu")

    # Return updated configuration
    return config


def h_threads(input_config, config):
    """Helper function to validate and setup thread configuration."""
    # Validate threads parameter is numeric
    if not isinstance(input_config["threads"], (int, float)):
        raise ValueError("ERR: threads must be a number")

    # Convert threads to integer
    threads = int(input_config["threads"])

    # Check BVM-specific thread limit
    if config["env"] == "bvm" and threads > 20:
        raise ValueError("ERR: BVM environment cannot request more than 20 threads")

    # Ensure minimum threads are left free
    if (os.cpu_count() - threads) < 2:
        raise ValueError(
            f"ERR: Must leave at least 2 threads free (requested {threads})"
        )

    # Set thread count in config with minimum of 1
    config["THREADS"] = max(1, threads)

    # Return updated configuration
    return config


def h_seed(input_config, config):
    """Helper function to set seed configuration."""
    # Set seed value in config
    config["seed"] = input_config["seed"]

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
    config.update(
        {
            "rows": input_config["rows"],
            "cols": input_config["cols"],
            "tokens": input_config["tokens"],
        }
    )

    return config


def h_data(config, input_config):
    """Helper function to setup data-related configurations."""
    ######## DATA DIR & DATASET ########
    if not os.path.isdir(input_config["data_dir"]):
        raise ValueError(
            f"ERR: data_dir '{input_config['data_dir']}' is not a valid path"
        )

    config.update(
        {"data_ds": input_config["data_ds"], "data_dir": input_config["data_dir"]}
    )

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

    # 1a. Approach: Short 3 low chars
    approach_str = config["approach"].lower()[:3]

    # 1b. Seed: Convert to string
    seed_str = str(config["seed"])

    # 1c. Environment: Short 1 low char
    env_str = config["env"].lower()[:1]

    # 1. First string
    first_str = approach_str + seed_str + env_str + "_"

    # 2a. Model Base: Mapped Short error checked
    modelbase_map = {"glove50": "g50", "bert-base-cased": "bbc"}
    if config["model_base"] in modelbase_map:
        modelbase_str = modelbase_map[config["model_base"]]
    else:
        raise ValueError(
            f"ERR: Model base '{config['model_base']}' not found in mappings"
        )

    # 2b. Model Name: Name of our defined class for model architecure
    modelname_str = config["model_name"]

    # 2. Second string
    second_str = modelbase_str + modelname_str + "_"

    # 3a. Data Set: low str
    ds_str = config["data_ds"]

    # 3b. Context: Rows+Cols+Tokens to str
    context_str = str(config["rows"]) + str(config["cols"]) + str(config["tokens"])

    # 3. Third string
    third_str = ds_str + context_str + "_"

    # 4a. Batch: ba followed by batch size
    batch_str = "ba" + str(config["batch_size"])

    # 4b. Learning Rate: lr followed by learning rate in scientific notation
    lr_str = "lr" + f"{config['lr']:.0e}".replace("e-0", "e-")

    # 4c. Epochs: ep followed by number of epochs
    epochs_str = "ep" + str(config["epochs"])

    # 4d. Patience: pa followed by patience value
    patience_str = "pa" + str(config["patience"])

    # 4. Fourth string
    fourth_str = batch_str + lr_str + epochs_str + patience_str + "_"

    # 5. Fifth string variable for approach, starting with vocab size
    fifth_str = "v" + str(config["vocab_size"] // 1000) + "k"

    # Construct base string based on these
    base_str = first_str + second_str + third_str + fourth_str + fifth_str

    # Define architecture specific strings for rnn/bert
    rnn_str = f"h{config['hidden_size']}l{config['num_hidden_layers']}"
    bert_str = f"i{config['intermediate_size']}a{config['num_attention_heads']}"

    # Map each approach to its corresponding augmented base string.
    approaches = {
        "simple": base_str,
        "rnn": base_str + rnn_str,
        "bert": base_str + rnn_str + bert_str,
        "saffu": base_str + "saffu_component",
    }

    # Set save_name based on the configuration's approach.
    save_name = approaches[config["approach"]]

    # Return final save name
    return save_name


def h_training(config, input_config):
    """Helper function to setup training parameters and call h_name for save name generation."""
    ######## TRAINING PARAMS ########
    config.update(
        {
            "batch_size": input_config["batch_size"],
            "lr": input_config["lr"],
            "mu": input_config["mu"],
            "epochs": input_config["epochs"],
            "patience": input_config["patience"],
            "save_int": input_config["save_int"],
            "save_dir": input_config["save_dir"],
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

    ######## ENVIRONMENT ########
    config = h_env(input_config)

    ######## DEVICE ########
    config = h_device(input_config, config)

    ######## THREADS ########
    config = h_threads(input_config, config)

    ######## SEED ########
    config = h_seed(input_config, config)

    ######## MODEL ########
    config = h_model(config, input_config)

    ######## DATA ########
    config = h_data(config, input_config)

    ######## APPROACH-SPECIFIC SETUP ########
    if config["approach"] in ["simple", "rnn"]:

        ######## VOCAB ########
        config = h_vocab(config, input_config)

        ######## SIMPLE LOADERS ########
        config = h_simpleloader(config)

        ######## RNN PARAMS ########
        if config["approach"] == "rnn":
            config = h_rnn(config, input_config)

    ######## BERT-SPECIFIC ########
    elif config["approach"] == "bert":

        ######## SETUP ########
        config = h_setupbert(config, input_config)

    ######## SAFFU-SPECIFIC ########
    elif config["approach"] == "saffu":
        pass

    ######## TRAINING & SAVE NAME ########
    config = h_training(config, input_config)

    return config


def display_config(config):
    """Display configuration in a readable format."""
    serializable_config = {
        k: (
            v
            if isinstance(v, (str, int, float, bool, type(None)))
            else f"<{v.__class__.__name__}>"
        )
        for k, v in config.items()
    }
    print(json.dumps(serializable_config, indent=2))
