# Import reload required funcs
import importlib
import itertools
import warnings

import openpyxl
import pandas as pd
import xlrd
from IPython.display import display
from xlrd.biffh import error_text_from_code
from xlrd.xldate import XLDateAmbiguous, XLDateError, xldate_as_tuple

from utils import parseutil

importlib.reload(parseutil)
from utils.parseutil import xls_content, xlsx_content, xlsx_dataType

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

    # Load the workbook and access the active sheet
    workbook = openpyxl.load_workbook(file_path, data_only=True)

    # Retrieve the sheet
    sheet = workbook.active

    # List to hold all cell contents
    sheet_contents = []

    # Iterate over all cells
    for row, col in itertools.product(range(sheet.max_row), range(sheet.max_column)):
        # Adjust to 1-based indexing
        cell = sheet.cell(row=row + 1, column=col + 1)

        # Extract cell type
        cell_type = xlsx_dataType(cell.value, cell.number_format)

        # Use the cell type to get value
        cell_content = xlsx_content(cell_type, cell)

        # Append content to the contents list
        sheet_contents.append(cell_content)

    # Return xlsx content
    return sheet_contents
