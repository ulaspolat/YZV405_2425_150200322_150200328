dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast
from transformers import XLMRobertaTokenizer
from typing import List, Dict, Tuple, Optional, Union

class IdiomDetectionDataset(Dataset):
    """
    Dataset for idiom detection task using tokenized sentences and idiom indices.
    """
    def __init__(self, file_path, tokenizer, max_length=128, language_filter=None):
        """
        Initialize the dataset from a CSV file.
        
        Args:
            file_path: Path to the CSV file with the dataset
            tokenizer: Tokenizer to tokenize the text
            max_length: Maximum sequence length
            language_filter: If provided, only load data for this language
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read the dataset
        self.data = pd.read_csv(file_path)
        
        # Filter by language if specified
        if language_filter:
            self.data = self.data[self.data['language'] == language_filter]
            
        if 'indices' in self.data.columns:
            self.has_labels = True
        else:
            self.has_labels = False
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a data example by index.
        
        Args:
            idx: Index of the example
        
        Returns:
            Dictionary with tokenized inputs and optionally labels
        """
        # Get text and possibly labels
        example = self.data.iloc[idx]
        text = example['sentence']
        language = example['language']
        example_id = example['id']
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get the input IDs and attention mask
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Prepare the result dictionary
        result = {
            'id': example_id,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'language': language,
            'text': text
        }
        
        # Add labels if available
        if self.has_labels:
            # Parse idiom indices from string representation
            idiom_indices = self._parse_idiom_indices(example['indices'])
            
            # Create token labels
            labels = self._create_token_labels(idiom_indices, len(input_ids))
            result['labels'] = labels
        
        return result
    
    def _parse_idiom_indices(self, label_str):
        """
        Parse idiom indices from string representation.
        
        Args:
            label_str: String representation of idiom indices
            
        Returns:
            List of integer indices
        """
        if isinstance(label_str, float) and pd.isna(label_str):
            return []
        
        label_str = label_str.strip('[]')
        if label_str == '' or label_str == '-1':
            return []
        
        return [int(idx) for idx in label_str.split(',')]
    
    def _create_token_labels(self, idiom_indices, seq_length):
        """
        Create token-level labels from idiom indices.
        
        Args:
            idiom_indices: List of token indices that are part of an idiom
            seq_length: Length of the tokenized sequence
            
        Returns:
            Binary tensor indicating which tokens are part of an idiom
        """
        labels = torch.zeros(seq_length)
        for idx in idiom_indices:
            if 0 <= idx < seq_length:
                labels[idx] = 1
        
        return labels
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of examples
            
        Returns:
            Dictionary with batched tensors
        """
        # Initialize empty lists for each key
        batch_dict = {
            'id': [],
            'input_ids': [],
            'attention_mask': [],
            'language': [],
            'text': []
        }
        
        if self.has_labels:
            batch_dict['labels'] = []
        
        # Collect tensors for each key
        for example in batch:
            for key in batch_dict:
                if key in example:
                    batch_dict[key].append(example[key])
        
        # Stack tensors where applicable
        batch_dict['input_ids'] = torch.stack(batch_dict['input_ids'])
        batch_dict['attention_mask'] = torch.stack(batch_dict['attention_mask'])
        
        if self.has_labels:
            batch_dict['labels'] = torch.stack(batch_dict['labels'])
        
        return batch_dict


def create_data_loaders(
    train_file: str,
    eval_file: str,
    tokenizer: XLMRobertaTokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    language_filter: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and evaluation.
    
    Args:
        train_file: Path to the training CSV file
        eval_file: Path to the evaluation CSV file
        tokenizer: Tokenizer to use
        batch_size: Batch size for the data loaders
        max_length: Maximum sequence length
        language_filter: If provided, only load data for this language
    
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    # Create datasets
    train_dataset = IdiomDetectionDataset(
        train_file,
        tokenizer,
        max_length=max_length,
        language_filter=language_filter
    )
    
    eval_dataset = IdiomDetectionDataset(
        eval_file,
        tokenizer,
        max_length=max_length,
        language_filter=language_filter
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, eval_dataloader 