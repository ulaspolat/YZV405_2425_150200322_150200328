# Idiom Detection Model

This repository contains a sophisticated idiom detection model for multilingual idiom detection in Turkish and Italian languages. The model can detect idioms in sentences and classify whether the idiom is used in its figurative (idiomatic) or literal sense.

## Architecture

The model uses a state-of-the-art architecture consisting of:

1. **XLM-RoBERTa Encoder**: Leverages a pretrained multilingual transformer model as the backbone for cross-lingual understanding.
2. **BiLSTM Layer**: Captures sequential information and context for better token classification.
3. **MLP Classifier**: Performs BIO (Beginning-Inside-Outside) sequence labeling.
4. **CRF Layer**: Improves prediction coherence by considering transition probabilities between labels.
5. **Class Weighting**: Handles class imbalance for better performance on minority classes.

## Dataset Structure

The model works with datasets containing:
- Sentences with idioms (in both idiomatic and literal usages)
- Tokenized versions of sentences
- Idiom expressions and their positions within sentences
- BIO tagging for token classification

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.10+
- Transformers 4.15+
- pandas, numpy, tqdm
- pytorch-crf

### Installation

```bash
git clone https://github.com/yourusername/idiom-detection.git
cd idiom-detection
pip install -r requirements.txt
```

## Usage with Google Colab

### Setup

1. Upload the `colab_files` directory to your Google Drive
2. Mount your Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Navigate to the project directory:
   ```python
   import os
   project_dir = '/content/drive/MyDrive/path/to/colab_files'
   os.chdir(project_dir)
   ```

### Training

Train the model for Turkish:

```python
!python train.py \
  --train_file /path/to/train.csv \
  --eval_file /path/to/eval.csv \
  --language_filter tr \
  --output_dir /path/to/models \
  --model_name "xlm-roberta-large" \
  --batch_size 16 \
  --num_epochs 15 \
  --use_crf True \
  --use_mlp True \
  --use_weighted_loss True \
  --seed 42
```

Train the model for Italian:

```python
!python train.py \
  --train_file /path/to/train.csv \
  --eval_file /path/to/eval.csv \
  --language_filter it \
  --output_dir /path/to/models \
  --model_name "xlm-roberta-large" \
  --batch_size 16 \
  --num_epochs 15 \
  --use_crf True \
  --use_bilstm True \
  --use_mlp True \
  --use_weighted_loss True \
  --seed 42
```

### Prediction

Generate predictions for Turkish:

```python
!python predict.py \
  --test_file /path/to/test.csv \
  --language_filter tr \
  --model_name "xlm-roberta-large" \
  --model_dir /path/to/models \
  --batch_size 16 \
  --output_dir /path/to/predictions \
  --seed 42
```

Generate predictions for Italian:

```python
!python predict.py \
  --test_file /path/to/test.csv \
  --language_filter it \
  --model_name "xlm-roberta-large" \
  --model_dir /path/to/models \
  --batch_size 16 \
  --output_dir /path/to/predictions \
  --seed 42
```

### Combining Predictions

```python
import pandas as pd

# Load language-specific predictions
tr_predictions = pd.read_csv('/path/to/tr_prediction.csv')
it_predictions = pd.read_csv('/path/to/it_prediction.csv')

# Combine predictions
combined_predictions = pd.concat([tr_predictions, it_predictions])
combined_predictions = combined_predictions.drop_duplicates(subset=['id'])

# Sort by ID
combined_predictions = combined_predictions.sort_values('id')

# Save combined predictions
combined_predictions.to_csv('/path/to/combined_prediction.csv', index=False)
```

## Model Components

### `model.py`

Contains the neural network architecture implementation with XLM-RoBERTa, BiLSTM, MLP, and CRF components. The model uses BIO tagging for sequence labeling and handles subword tokenization for accurate idiom detection.

### `dataset.py`

Implements dataset handling and preprocessing for idiom detection, including:
- Converting words to subwords
- BIO tagging for sequence labeling
- Calculating class weights for imbalance handling
- Batch preparation and collation

### `train.py`

Handles model training with features like:
- Learning rate scheduling
- Early stopping based on F1 score
- Gradient clipping
- Class weighting for imbalanced data
- Reproducibility through seed setting

### `predict.py`

Generates predictions from trained models with:
- Language-specific model loading
- Subword to word index conversion
- Proper handling of BIO tagging
- F1 score evaluation during prediction

## Parameter Tuning

For optimal results, experiment with these hyperparameters:

- `--batch_size`: 8, 16, 32 (larger batch sizes require more GPU memory)
- `--learning_rate`: 1e-5 to 5e-5 
- `--num_epochs`: 5-15 for typical datasets
- `--max_length`: 128 is sufficient for most sentences, 256 for longer ones
- `--dropout_rate`: 0.1-0.3 for regularization
- `--lstm_hidden_dim`: 256, 512, 768
- `--mlp_hidden_dim`: 256, 512, 768

## Performance

With the full architecture (XLM-RoBERTa, BiLSTM, MLP, CRF) and proper hyperparameter tuning, the model achieves strong performance on idiom detection in both Turkish and Italian languages. Class weighting significantly improves detection of idiomatic usages. When tested on unseen sentences containing the same idioms from our dataset, the model achieved the following F1 scores:
- Turkish (F1-TR): 0.9224
- Italian (F1-IT): 0.9104
- Average F1: 0.9164

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This model builds on the XLM-RoBERTa multilingual transformer from Hugging Face Transformers library. 
