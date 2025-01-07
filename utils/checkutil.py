import importlib
import sys
import os
import torch
import pandas as pd
from IPython.display import display


def observe_datadir(data_dirs):
    """
    Compare file counts across multiple data directories and flag rows with differing counts.
    Args:
        data_dirs (list): List of directory paths to compare.
    """
    # Dictionary to store subdirectory counts for each directory
    results = {}

    for data_dir in data_dirs:
        # Extract the last level of the data directory path
        dir_name = os.path.basename(os.path.normpath(data_dir))

        # Initialize a dictionary to store subdirectory names and file counts
        subdir_file_counts = {}

        # Loop through the subdirectories and count the files
        for subdir, _, files in os.walk(data_dir):
            # Skip the root directory itself
            if subdir != data_dir:
                subdir_name = os.path.basename(subdir)  # Get the subdirectory name
                subdir_file_counts[subdir_name] = len(files)  # Count the number of files

        # Store the results in the main dictionary
        results[dir_name] = subdir_file_counts

    # Get the union of all subdirectory names across all directories
    all_subdirs = set()
    for subdir_counts in results.values():
        all_subdirs.update(subdir_counts.keys())

    # Custom sort function for subdirectories
    def custom_sort_key(subdir_name):
        # Check if the name starts with train, val, or test
        prefixes = ("train_", "val_", "test_")
        if not subdir_name.startswith(prefixes):
            # Directories without train/val/test prefixes come first
            return (0, subdir_name)
        else:
            # Split prefix and suffix
            prefix, suffix = subdir_name.split("_", 1)
            prefix_order = {"train": 1, "val": 2, "test": 3}
            return (1, suffix, prefix_order.get(prefix, 4))

    # Sort the subdirectories using the custom sort key
    sorted_subdirs = sorted(all_subdirs, key=custom_sort_key)

    # Initialize a DataFrame with all subdirectory names and directory names as columns
    df = pd.DataFrame(
        columns=[os.path.basename(os.path.normpath(d)) for d in data_dirs]
    )
    df["Subdirectory"] = sorted_subdirs
    df.set_index("Subdirectory", inplace=True)

    # Populate the DataFrame with file counts, filling missing values with zero
    for dir_name, subdir_counts in results.items():
        df[dir_name] = df.index.map(subdir_counts).fillna(0).astype(int)

    # Add the "diff" column to flag rows with differing counts
    df["diff"] = df.nunique(axis=1).apply(lambda x: "Y" if x > 1 else "")

    # Set pandas options to display all rows and columns
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    # Display the DataFrame
    display(df)