# Import reload required funcs
import csv
import importlib
import itertools
import os
import warnings

import openpyxl
import pandas as pd
import xlrd
from IPython.display import display
from joblib import Parallel, delayed
from tqdm import tqdm
from xlrd.biffh import error_text_from_code
from xlrd.xldate import XLDateAmbiguous, XLDateError, xldate_as_tuple

# Custom imports
from utils import parseutil, selfutil

importlib.reload(parseutil)
importlib.reload(selfutil)
from utils.parseutil import xls_content, xlsx_content, xlsx_dataType
from utils.selfutil import get_fileList

"""
@==========================================================================@
                              1. TOKENIZER 
@==========================================================================@
"""


# [1a] Tokenizer Data for XLS files
def xls_tokenize(file_path):

    # Open the workbook and get the first sheet
    workbook = xlrd.open_workbook(filename=file_path, formatting_info=True)

    # List to hold all cell contents
    sheet_contents = []

    df_dict = pd.read_excel(
        file_path,
        header=None,
        dtype=str,
        na_values=" ",
        keep_default_na=False,
        sheet_name=None,
        engine="xlrd",
    )

    # Loop over all sheets in the workbook
    for sheet_index, sheet in enumerate(workbook.sheets()):

        # Get the matching DataFrame for this sheet
        df_sheet = list(df_dict.values())[sheet_index]

        # Iterate over each cell using itertools.product
        for row_idx, col_idx in itertools.product(
            range(sheet.nrows), range(sheet.ncols)
        ):

            # Get the cell object
            cell = sheet.cell(row_idx, col_idx)

            # Get type and content using the helper function
            _, cell_content = xls_content(cell, workbook, df_sheet, row_idx, col_idx)

            # Append content to the contents list
            sheet_contents.append(cell_content)

    # Remove duplicates since tokenizer training
    sheet_contents = list(set(sheet_contents))

    # Return xls content
    return sheet_contents


# [1b] Tokenizer Data for XLSX files
def xlsx_tokenize(file_path):

    # Suppress all UserWarnings related to various xlsx issues
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load the workbook with formulas resolved
    workbook = openpyxl.load_workbook(file_path, data_only=True)

    # Flat list to hold all cell contents across all sheets
    sheet_contents = []

    # Loop through all sheets
    for sheet in workbook.worksheets:

        # Iterate over all cells in the current sheet
        for row, col in itertools.product(
            range(sheet.max_row), range(sheet.max_column)
        ):

            # Adjust to 1-based indexing (openpyxl requirement)
            cell = sheet.cell(row=row + 1, column=col + 1)

            # Extract cell type
            cell_type = xlsx_dataType(cell.value, cell.number_format)

            # Use the cell type to get value
            cell_content = xlsx_content(cell_type, cell)

            # Append content to the flat list
            sheet_contents.append(cell_content)

    # Return deduplicate
    return list(set(sheet_contents))


# [1c] Tokenizer Data for CSV files
def csv_tokenize(file_path):

    # List to hold all cell contents
    sheet_contents = []

    # Get the delimiter from the file
    with open(file_path, mode="r", newline="", encoding="utf-8") as csvfile:
        delimiter = csv.Sniffer().sniff(csvfile.read()).delimiter

    # Go through the CSV file and read each row
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=delimiter)
        for row in reader:
            for cell in row:
                sheet_contents.append(cell)

    # Deduplicate the contents and return
    return list(set(sheet_contents))


def _process_single_file_for_tokenizer(file):
    """Process a single file for tokenizer data.

    Args:
        file (str): Path to the file to process.

    Returns:
        list: List of unique cell contents from the file.
    """
    try:
        # Get the file extension
        file_ext = file.split(".")[-1].lower()

        # Check the file type and call the appropriate function
        if file_ext == "xls":
            sheet_contents = xls_tokenize(file)
        elif file_ext == "xlsx":
            sheet_contents = xlsx_tokenize(file)
        elif file_ext == "csv":
            sheet_contents = csv_tokenize(file)
        else:
            print(f"Unsupported file type: {file_ext}")
            return []

        return sheet_contents

    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")
        return []


# [1d] Final func to take dir and return all its sheets' data
def get_saffutok_data(dir_path, threads=1):

    # Get the list of valid/invalid files
    files, _ = get_fileList(dir_path)

    try:
        # Process files in parallel and get a list of lists
        results = Parallel(n_jobs=max(1, threads), timeout=99999)(
            delayed(_process_single_file_for_tokenizer)(f)
            for f in tqdm(files, desc="Tokenizing files", unit="file")
        )

        # Flatten all results directly and deduplicate in one go
        all_contents = set(
            content for file_contents in results for content in file_contents
        )

        # Print information
        print(f"Files/Tokens: {len(files)}/{len(all_contents)}")

        return list(all_contents)

    except Exception as e:
        raise e
