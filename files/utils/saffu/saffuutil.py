import importlib

# Import the utilities and the dataloader
from utils import selfutil
from classes import SAFFUDataLoader

# Now reload the modules to ensure they are up-to-date
importlib.reload(selfutil)
importlib.reload(SAFFUDataLoader)

# Import the funcs needed from utils
from utils.selfutil import spreadsheet_to_df

# Import the SAFFUDataLoader class
from classes.SAFFUDataLoader import SAFFUDataLoader

# Other regular imports
import torch.nn as nn
import torch
from tqdm import tqdm_notebook as tqdm
from tqdm_joblib import tqdm_joblib
import gc
import os
import pandas as pd
import math
import time
import sys
from joblib import Parallel, delayed
import random

def saffutok_traindata(data_dir):
    """
    Processes files in the specified directory in parallel, extracting unique text entries
    from each file to build a consolidated set of unique texts.

    Args:
        data_dir (str): The directory containing the files to be processed.

    Returns:
        list: A list containing all unique text entries across all files.
    """
    
    def process_file(file_path):
        """
        Extracts unique text entries from a single file.

        Args:
            file_path (str): The path to the file to be processed.

        Returns:
            set: A set of unique text entries from the file, or an empty set if an error occurs.
        """
        # Convert the spreadsheet to DataFrame and get unique values as a set else return empty set
        try:
            return spreadsheet_to_df(file_path, 0, 0).applymap(str).values.flatten().tolist()
        except Exception as e:
            return []

    # Ensure the provided directory exists
    if not os.path.isdir(data_dir):
        raise ValueError(f"{data_dir} NOT FOUND")

    # Retrieve all valid file paths
    file_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) 
                  if filename.lower().endswith(('.xls', '.xlsx', '.csv')) and os.path.isfile(os.path.join(data_dir, filename))]


    results = Parallel(n_jobs=os.cpu_count() // 2)(
        delayed(process_file)(file_path) for file_path in tqdm(file_paths, desc="PARA - Generating Docs")
    )
    
    # Return consolidated unique texts from each file as list
    return [item for sublist in results for item in sublist]

# Function to load/train saffu tokenizer given a predefined one
def load_saffutok(reload, vocab_file, tokenizer, tokenizer_name, tokenizer_directory, train_dir='../data/train/'):
    
    # If we are not reloading and the vocab_file path exists then
    if not reload and os.path.exists(vocab_file):

        # Print message for loading tokenizer
        print(f"Loading tokenizer: {tokenizer_name}\n")

        # Store the loaded tokenizer from the directory into result
        result = tokenizer.load(tokenizer_name, load_directory=tokenizer_directory)

    # If we are either reloading or the vocab_file path doesn't exist then    
    else:
        
        # Print message that docs is being prepared
        print(f'Preparing docs')
        
        # Run the func to get list
        docs = saffutok_traindata(train_dir)

        # Print the training message
        print(f"Training tokenizer: {tokenizer_name}")

        # Train our tokenizer
        tokenizer.train(tokenizer.pretokenize_documents(docs))

        # Print saving message
        print(f"Saving vocabulary for {tokenizer_name} tokenizer at {tokenizer_directory}")

        # Save the vocabulary in the directory specified
        tokenizer.save_vocabulary(tokenizer_name, save_directory=tokenizer_directory)
        

        
# Func to convert spreadsheet to 2D list convo format
def spread2convo(file_path):
    return [["Cell", str(cell)] for row in spreadsheet_to_df(file_path).values for cell in row]
    

# Define func to go through all the files in directory and return convos format
def dir2convos(data_dir):
    
    # Empty list to store all convos
    convos = []
    
    # Retrieve all the valid file_paths
    file_paths = [ os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if
                  filename.lower().endswith(('.xls', '.xlsx', '.csv')) and 
                  os.path.isfile(os.path.join(data_dir, filename)) ]
    
    # Go through all the files in directory
    for file in tqdm(file_paths, desc = 'Getting convos list'):
        
        # Try catch block for unreadable files
        try:
            
            # Read the file in and extract list then append to list of all convos
            convos.append(spread2convo(file))
        
        # Exception means file couldnt be parsed
        except Exception as e:
            
            # Print path and error
            print(f'ERROR {file}: {e}')
    
    # Return the full convos list finally
    return convos  

def get_saffuloader(data_dir, tokenizer, length = None):

    # Gather all file paths with specified extensions
    file_paths = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.lower().endswith(('.xls', '.xlsx', '.csv')) and os.path.isfile(os.path.join(data_dir, filename))
    ]

    # If length is specified and less than available file_paths, randomly sample from file_paths
    if length is not None and length < len(file_paths):
        random.seed(42)  # Set a seed for reproducibility
        file_paths = random.sample(file_paths, length)

    # Return the SAFFUDataLoader object with the filtered or complete list
    return SAFFUDataLoader(file_paths, tokenizer)
