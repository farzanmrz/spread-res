# CHANGELOG

All notable changes to this project will be documented in this file.

## [Initial Documentation] - 2025-07-02 14:30:00

### Added
- **Project Overview**: Spreadsheet Cell Formatting Detection System
  - BERT-based machine learning pipeline for predicting cell formatting (specifically bold cells) in spreadsheet data
  - System processes Excel files and uses deep learning to classify which cells should be bold based on content and context

### Pipeline Architecture
- **Setup & Configuration** (`setuputil.py:498`)
  - Environment detection: Colab/local/BVM with appropriate device (CUDA/MPS/CPU)
  - Configuration validation: Validates BERT architecture params, training settings
  - Data directory setup: Creates paths for train/val/test splits from `data_ds="50"`

- **Data Loading** (`classes/Loader.py:174`)
  - LoaderBert class: Processes Excel files in parallel using HuggingFace tokenizer
  - Spreadsheet parsing: Extracts cell content, formatting metadata via `parseutil.process_spreadsheet`
  - Tokenization: Converts cell text to BERT tokens (100x100 grid, 32 tokens per cell)
  - Label extraction: Bold formatting stored in `y_tok[:,:,6]` tensor dimension

- **Model Architecture** (`classes/models/BertGrid.py:11`)
  - Dual BERT structure:
    - `bertModel_cell`: Processes individual cell content (pretrained bert-base-cased or custom)
    - `bertEncoder_spatial`: Learns spatial relationships between cells
  - Positional encoding: Custom row/col position embeddings for grid structure
  - Binary classification: Final layer predicts bold probability per cell

- **Training Process** (`utils/trainutil.py:15`)
  - Loss function: BCEWithLogitsLoss with class imbalance handling
  - Optimization: AdamW optimizer with gradient clipping
  - Early stopping: Monitors validation perplexity with patience=2
  - Model saving: Periodic saves every 5 epochs to `../models/`

- **Inference & Evaluation** (`utils/inferutil.py`)
  - Single inference (`infer_one:57`): Tests individual spreadsheets with metrics
  - Batch inference (`infer_full:248`): Evaluates entire datasets
  - Metrics calculation: Accuracy, precision, recall, F1-score, confusion matrices
  - Visualization: Displays predicted vs actual bold cell grids

### Key Components
- **`setup_config()`**: Master configuration orchestrator calling 11 helper functions
- **`train_model()`**: Unified training loop handling BERT-specific forward passes
- **`get_logitlabels()`**: Extracts predictions and labels for loss calculation
- **`process_spreadsheet()`**: Core data processing from parseutil (converts Excel to tensors)
- **`BertGrid.forward()`**: Complex forward pass combining cell encoding + spatial reasoning

### Performance
- Training completed: 16 epochs with early stopping
- Convergence: Validation loss improved from 1.39 to 0.093
- Architecture: Combines content understanding (cell-level BERT) with spatial context (grid-level BERT encoder)


Regarding data all 18 types of metadata from y tok will be one hot encoded

The 100 x 100 x 180 metadata and mask is exact same shape

Inside train loop we loop through each type of metadata and 3rd dim the location of OHE vectors from start to finish is fed in so that can be masked inside the model's forward

The loss function in training loop changes from BCEWithLogitsLoss to CrossEntropyLoss, this takes basically the number of logits as many as there are number of classes. So binary this will take 2, if 13 classes then give 13 logits

Speaking of model, inside the layer you have named binary classifier in the end the linear layer is the output head. **One output head per metadata task will exist**, the current one is projecting to 1 for binary change to 2 since binary, one per each type of metadata final dim is as many number of classes there are.