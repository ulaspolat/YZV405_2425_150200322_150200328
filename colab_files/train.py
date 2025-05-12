import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer, get_linear_schedule_with_warmup, set_seed as set_transformers_seed
from tqdm import tqdm
import numpy as np
import pandas as pd
import random  # Added for reproducibility

from model import IdiomDetectionModel
from dataset import create_data_loaders

# Function to set random seeds for reproducibility
def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Additional deterministic settings for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set Python hash seed for reproducibility of operations like dictionary iteration
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set seed for the transformers library
    set_transformers_seed(seed)

# Function to calculate per-row F1 score like in scoring.py
def calculate_f1_score(predicted_indices_list, gold_indices_list):
    f1_scores = []

    # Compute F1 scores for each row
    for pred, gold in zip(predicted_indices_list, gold_indices_list):
        # Special case for ground truth [-1]
        if gold == [-1]:
            # Prediction must also be exactly [-1]
            if pred == [-1]:
                f1_scores.append(1.0)
            else:
                f1_scores.append(0.0)
        else:
            # Convert indices into sets for comparison
            pred_set = set(pred) if isinstance(pred, list) else set()
            gold_set = set(gold) if isinstance(gold, list) else set()

            # Calculate precision, recall, and F1 score
            intersection = len(pred_set & gold_set)
            precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
            recall = intersection / len(gold_set) if len(gold_set) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)

    # Compute mean F1 score
    return np.mean(f1_scores)

def train(args):
    """
    Train the model on the idiom detection task.
    
    Args:
        args: Command-line arguments
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Check if we're running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        print("Running in Google Colab environment")
    
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print info about CUDA for debugging
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    
    # Create data loaders
    train_dataloader, eval_dataloader = create_data_loaders(
        args.train_file,
        args.eval_file,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        language_filter=args.language_filter,
        seed=args.seed
    )
    
    print(f"Train dataset size: {len(train_dataloader.dataset)}")
    print(f"Eval dataset size: {len(eval_dataloader.dataset)}")
    
    # Create model
    model = IdiomDetectionModel(args.model_name, args.dropout_rate)
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * args.num_epochs
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0.0
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_dataloader, desc="Training")
        
        for batch in train_pbar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            logits, loss = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation
        model.eval()
        eval_loss = 0.0
        all_predicted_indices = []
        all_gold_indices = []
        
        eval_pbar = tqdm(eval_dataloader, desc="Evaluating")
        for batch in eval_pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.no_grad():
                logits, loss = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            
            eval_loss += loss.item()
            
            # Convert logits to idiom indices for each example in the batch
            predicted_indices = model.convert_logits_to_indices(
                logits, 
                batch['attention_mask']
            )
            all_predicted_indices.extend(predicted_indices)
            
            # Generate gold indices from labels for each example
            for i in range(batch['labels'].size(0)):
                # Get positions where label is 1 (idiom token)
                gold_idx = torch.where(batch['labels'][i] == 1)[0].cpu().tolist()
                # If no idiom tokens, set to [-1]
                if len(gold_idx) == 0:
                    gold_idx = [-1]
                all_gold_indices.append(gold_idx)
        
        avg_eval_loss = eval_loss / len(eval_dataloader)
        
        # Calculate F1 score using the same logic as in scoring.py
        f1 = calculate_f1_score(all_predicted_indices, all_gold_indices)
        
        print(f"Average evaluation loss: {avg_eval_loss:.4f}")
        print(f"F1 score: {f1:.4f}")
        
        # Save the best model
        if f1 > best_f1:
            best_f1 = f1
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            
            model_path = os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}-{args.language_filter}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
    
    print(f"Best F1 score: {best_f1:.4f}")
    return best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for idiom detection")
    
    # Data arguments - update default paths for Google Drive structure
    parser.add_argument("--train_file", type=str, default="/content/drive/MyDrive/nlp-project/public_data/train.csv", help="Path to the training file")
    parser.add_argument("--eval_file", type=str, default="/content/drive/MyDrive/nlp-project/public_data/eval.csv", help="Path to the evaluation file")
    parser.add_argument("--language_filter", type=str, default=None, help="Filter for a specific language (e.g., 'tr')")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/nlp-project/models", help="Output directory for saved models")
    
    # Add seed argument for reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="xlm-roberta-large", help="Name of the pretrained model")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the classification head")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for the scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    
    args = parser.parse_args()
    train(args) 