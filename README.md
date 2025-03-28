# 4D contextual transfomer for tabular data

## Project Overview
This project investigates using 4D context transformers for spreadsheet analysis. The primary task is supervised learning to predict whether cells are bold or non-bold based on surrounding context and cell content. The architecture uses specialized BERT-based models adapted for grid-structured data with custom positional encodings.

## Project Structure
- `classes/models/`: Model architectures
  - `BertGrid.py`: BERT-based model with custom spatial encoding for grid data
  - `BertPreTinyNew.py`: Optimized BERT variant with pretrained weights and spatial awareness
  - `Rnn2d.py`, `SimpleGeluEmbed.py`: Alternative model architectures (older models)
- `classes/`: Core dataset classes
  - `Loader.py`: Custom PyTorch DataLoader optimized for spreadsheet processing
  - `Vocab.py`: Legacy vocabulary class for non-BERT approaches
- `utils/`: Helper functions
  - `trainutil.py`: Training loop and optimization functions
  - `inferutil.py`: Inference, evaluation and visualization tools
  - `parseutil.py`: Spreadsheet parsing and preprocessing utilities
  - `setuputil.py`: Environment and configuration setup
- `data/`: Dataset directories with pre-processed spreadsheets
  - Organized by size (50, 100, 250, 500, 1k, 2k, all) and split (train, val, test)
- `runscr/`: Training and execution notebooks
  - `cBertGrid_50.ipynb`: Complete pipeline for BertGrid model training
  - `cBertPreTiny_50.ipynb`, `cBertPreTiny_1k.ipynb`: BertPreTinyNew model training
  - `test_devModels.ipynb`: Experimentation notebook for model development
- `testscr/`: Testing and verification notebooks
- `models/`: Saved model checkpoints with timestamp-based naming

## Data Pipeline
1. **Data Loading**: `Loader.py` classes handle spreadsheet parsing through:
   - `LoaderBert`: For BERT-based tokenization with HuggingFace tokenizers
   - `LoaderSimple`: For vocabulary-based tokenization (legacy)
2. **Preprocessing**: `parseutil.py` extracts text and metadata (including bold formatting)
   - Padding/truncating to fixed dimensions (default: 100x100 cells, 32 tokens per cell)
   - Parallel processing for efficiency using joblib
3. **Input Representation**:
   - Each cell: Token IDs (x_tok) and attention masks (x_masks)
   - Target labels: Metadata matrix with bold info at index 6 (y_tok[:,:,:,6])

## Models
1. **BertGrid**: 
   - Two-stage BERT architecture applying pooled cell encodings to a spatial encoder
   - Custom positional encoding for grid data (row/column awareness)
   - Final binary classifier to predict bold status

2. **BertPreTinyNew**:
   - Uses pretrained BERT weights for efficiency
   - Projects to architecture-specific embedding dimensions
   - Enhanced positional encoding optimized for grid structure

## Training Commands
- Run BertGrid training: `jupyter notebook runscr/cBertGrid_50.ipynb`
- Run BertPreTiny training:
  - Small dataset: `jupyter notebook runscr/cBertPreTiny_50.ipynb`
  - Larger dataset: `jupyter notebook runscr/cBertPreTiny_1k.ipynb`
- Model development testing: `jupyter notebook runscr/test_devModels.ipynb`

## Evaluation Methods
- **Single example**: `infer_one()` in inferutil.py - displays prediction details for a single spreadsheet
- **Full evaluation**: `infer_full()` in inferutil.py - processes multiple examples with metrics
- Confusion matrices and cell visualizations are auto-generated
- Metrics: Accuracy, Precision, Recall, F1-score with special focus on bold cells

## Code Style Guidelines
- **Imports**: Group by category (standard library → third-party → project modules)
- **Model definitions**:
  - Initialize with config dict containing hyperparameters
  - Document internal components with inline comments
  - Use descriptive variable names (esp. for tensors)
- **Comments**: Major functional sections with header comments
- **Error handling**: Use try/except blocks with specific exception types
- **Function clarity**: Prefer multiple short functions over monolithic implementations
- **Tensor operations**: Optimize for memory efficiency, especially when using GPU

## Configuration Standards
Configuration follows a hierarchical approach using dictionaries:
```python
config = {
    # Environment settings
    "env": "colab",  # or "local"
    "approach": "bert",  # or "simple"
    
    # Model parameters
    "model_name": "BertGrid",  # or "BertPreTinyNew"
    "hidden_size": 128,
    "num_hidden_layers": 2,
    
    # Data parameters
    "rows": 100,
    "cols": 100,
    "tokens": 32,
    
    # Training parameters
    "batch_size": 8,
    "lr": 5e-5,
    "patience": 5
}
```

## Imbalance Handling
- Dataset typically has extreme imbalance (many non-bold cells, few bold cells)
- `BCEWithLogitsLoss` with positive weighting based on dataset imbalance ratio
- Evaluation metrics prioritize precision/recall over simple accuracy
- Use `get_imbalance()` method from Loader class to compute class weighting

## Troubleshooting
- CUDA memory issues: Reduce batch size or model complexity
- Preprocessing errors: Check logs for file paths in `failed_files` list
- Poor performance: Examine distribution of bold cells in dataset
- Out-of-distribution data: Run inferutil's infer_one with disp_sig=True for detailed analysis