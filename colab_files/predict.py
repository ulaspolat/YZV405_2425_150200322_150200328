import os
import argparse
import torch
import pandas as pd
from transformers import XLMRobertaTokenizer
from tqdm import tqdm
import ast

# Update the import paths to use absolute imports
from model import IdiomDetectionModel
from dataset import IdiomDetectionDataset

def predict(args):
    """
    Generate predictions for a test file using a trained model.
    
    Args:
        args: Command-line arguments
    """
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
    
    # Load the test data
    test_dataset = IdiomDetectionDataset(
        args.test_file,
        tokenizer,
        max_length=args.max_length,
        language_filter=args.language_filter
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Initialize models for each language
    models = {}
    
    # Load the model for each language
    languages = [args.language_filter] if args.language_filter else ['tr', 'it']
    
    for lang in languages:
        # Create the model
        model = IdiomDetectionModel(args.model_name, args.dropout_rate)
        
        # Load the trained model
        model_path = os.path.join(args.model_dir, f"{args.model_name.split('/')[-1]}-{lang}.pt")
        if os.path.exists(model_path):
            print(f"Loading model for language {lang} from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Warning: Model for language {lang} not found at {model_path}. Using default parameters.")
        
        model.to(device)
        model.eval()
        models[lang] = model
    
    # Generate predictions
    predictions = []
    ids = []
    languages_list = []
    
    for batch in tqdm(test_dataloader, desc="Generating predictions"):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        batch_languages = batch.get('language', [args.language_filter] * len(batch['input_ids']))
        
        # Process each example in the batch based on its language
        for i in range(len(batch['input_ids'])):
            example = {k: v[i].unsqueeze(0) if isinstance(v, torch.Tensor) else v[i] for k, v in batch.items()}
            lang = example.get('language', args.language_filter)
            
            # Use the corresponding model for this language
            if lang in models:
                model = models[lang]
            else:
                # Default to the first model if language not found
                model = list(models.values())[0]
            
            # Generate prediction
            with torch.no_grad():
                logits, _ = model(
                    input_ids=example['input_ids'],
                    attention_mask=example['attention_mask']
                )
            
            # Convert logits to indices
            indices = model.convert_logits_to_indices(logits, example['attention_mask'])
            
            # Store prediction
            predictions.append(indices[0])
            ids.append(example['id'].item())
            languages_list.append(lang)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'id': ids,
        'language': languages_list,
        'indices': [str(p) for p in predictions]
    })
    
    # Sort by ID
    output_df = output_df.sort_values('id')
    
    # Save predictions
    output_path = os.path.join(args.output_dir, 'prediction.csv')
    output_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for idiom detection")
    
    # Data arguments - update paths for Google Drive
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test file")
    parser.add_argument("--language_filter", type=str, default=None, help="Filter for a specific language (e.g., 'tr')")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/nlp-project/predictions", help="Output directory for predictions")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="xlm-roberta-large", help="Name of the pretrained model")
    parser.add_argument("--model_dir", type=str, default="/content/drive/MyDrive/nlp-project/models", help="Directory containing trained models")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the classification head")
    
    # Prediction arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction")
    
    args = parser.parse_args()
    predict(args) 