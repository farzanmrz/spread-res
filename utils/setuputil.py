# General imports
import os
import torch
import importlib
import copy
import json
from transformers import AutoTokenizer

# Reload the selfutil module and import required functions
from utils import selfutil
from classes import Loader

importlib.reload(selfutil)
importlib.reload(Loader)

# Import utils and classes needed
from utils.selfutil import set_seed, get_vocab, create_embeddings, get_fileList
from classes.Loader import LoaderSimple, LoaderBert


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

    config.update({"env": input_config["env"], "approach": input_config["approach"]})

    ######## DEVICE ########
    device_config = input_config["device"]
    if (
        device_config.startswith("cuda")
        and torch.cuda.is_available()
        and int(device_config.split(":")[1]) < torch.cuda.device_count()
    ):
        config["DEVICE"] = torch.device(device_config)
    elif (
        device_config.startswith("mps")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        config["DEVICE"] = torch.device("mps")
    else:
        config["DEVICE"] = torch.device("cpu")

    ######## THREADS ########
    if not isinstance(input_config["threads"], (int, float)):
        raise ValueError("ERR: threads must be a number")

    threads = int(input_config["threads"])
    if (os.cpu_count() - threads) < 2:
        raise ValueError(
            f"ERR: Must leave at least 2 threads free (requested {threads})"
        )
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
            "hidden_dim": input_config.get("hidden_dim", 128),
            "rnn_layers": input_config.get("rnn_layers", 2),
            "dropout_rate": input_config.get("dropout_rate", 0.05),
            "nonlinearity": input_config.get("nonlinearity", "relu"),
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

    # Construct base string based on these
    base_str = first_str + second_str + third_str + fourth_str

    # If approach is simple
    if config["approach"].lower() == "simple":

        # 1a. Vocab case: First char of case
        vcase_str = config["vocab_case"][0]

        # 1b. Vocab space: Sp if True, Nsp if False
        vspace_str = "Sp" if config["vocab_space"] else "Nsp"

        # 1c. Vocab size: k suffix for thousands
        vsize_str = str(config["vocab_size"] // 1000) + "k"

        # 1. Vocab string
        vocab_str = vcase_str + vspace_str + vsize_str

        # Set the save name
        save_name = base_str + vocab_str

    # If approach is rnn
    elif config["approach"].lower() == "rnn":

        # 1a. Vocab case: First char of case
        vcase_str = config["vocab_case"][0]

        # 1b. Vocab space: Sp if True, Nsp if False
        vspace_str = "Sp" if config["vocab_space"] else "Nsp"

        # 1c. Vocab size: k suffix for thousands
        vsize_str = str(config["vocab_size"] // 1000) + "k"

        # 1. Vocab string
        vocab_str = vcase_str + vspace_str + vsize_str + "_"

        # 2a. Hidden Dim: hid followed by hidden dimension
        hdim_str = "h" + str(config["hidden_dim"])

        # 2a. RNN Layers: rnn followed by number of layers
        layer_str = "l" + str(config["rnn_layers"])

        # 2. RNN string
        rnn_str = hdim_str + layer_str

        # Set the save name
        save_name = base_str + vocab_str + rnn_str

    # If approach is bert
    elif config["approach"].lower() == "bert":

        # 1a. Vocab size: v followed by vocab size
        vsize_str = "v" + str(config["vocab_size"])

        # 1b. Hidden size: h followed by hidden size
        hsize_str = "h" + str(config["hidden_size"])

        # 1c. Intermediate size: i followed by intermediate size
        isize_str = "i" + str(config["intermediate_size"])

        # 1d. Hidden Layers: l followed by number of layers
        hlayer_str = "l" + str(config["num_hidden_layers"])

        # 1e. Attention Heads: a followed by number of heads
        ahead_str = "a" + str(config["num_attention_heads"])

        # 1. BERT string
        bert_str = vsize_str + hsize_str + isize_str + hlayer_str + ahead_str

        # Set the save name
        save_name = base_str + bert_str

    # If approach is saffu
    elif config["approach"].lower() == "saffu":
        # SAFFU-specific naming logic
        save_name = base_str + "saffu_component"

    # Return the save name
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
