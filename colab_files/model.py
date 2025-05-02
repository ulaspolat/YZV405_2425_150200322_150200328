import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from typing import Optional, Tuple, List

class IdiomDetectionModel(nn.Module):
    """
    Idiom Detection Model using XLM-RoBERTa-large backbone.
    This model predicts whether each token is part of an idiom (B, I) or not (O).
    """
    def __init__(self, model_name, dropout_rate=0.1):
        """
        Initialize the idiom detection model.
        
        Args:
            model_name: Name of the pretrained model to use as the encoder
            dropout_rate: Dropout rate for the classification head
        """
        super().__init__()
        
        # Load model configuration
        self.config = XLMRobertaConfig.from_pretrained(model_name)
        
        # Load pre-trained encoder
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor indicating which tokens are padding
            labels: Optional tensor of token labels
            
        Returns:
            logits: Tensor of logits for each token
            loss: Loss value if labels are provided, otherwise None
        """
        # Get outputs from the encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get the last hidden state
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout and classification
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        # Squeeze the last dimension to get shape [batch_size, seq_len]
        logits = logits.squeeze(-1)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
        
        return logits, loss
    
    def convert_logits_to_indices(self, logits, attention_mask, threshold=0.5):
        """
        Convert logits to idiom indices.
        
        Args:
            logits: Tensor of logits with shape [batch_size, seq_len]
            attention_mask: Tensor indicating which tokens are padding
            threshold: Threshold for binary classification
            
        Returns:
            List of lists, where each inner list contains the indices of tokens
            predicted as part of an idiom for the corresponding example, or [-1] if no idiom found
        """
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits)
        
        # Apply threshold for binary predictions
        binary_predictions = (probabilities > threshold).float()
        
        # Use attention mask to filter out padding tokens
        masked_predictions = binary_predictions * attention_mask
        
        # Convert to idiom indices
        batch_idiom_indices = []
        for i in range(masked_predictions.size(0)):
            idiom_indices = torch.where(masked_predictions[i] == 1)[0].tolist()
            # If no idiom found, return [-1] instead of empty list
            if len(idiom_indices) == 0:
                idiom_indices = [-1]
            batch_idiom_indices.append(idiom_indices)
        
        return batch_idiom_indices
    
    def convert_indices_to_labels(self, indices_list: List[List[int]], seq_length: int) -> torch.Tensor:
        """
        Convert idiom indices to BIO labels.
        
        Args:
            indices_list: List of lists where each inner list contains indices of idiom tokens
            seq_length: Length of the sequence
            
        Returns:
            Tensor of BIO labels for each token in the sequence
        """
        # Create a tensor initialized with O labels (class 2)
        labels = torch.full((len(indices_list), seq_length), 2, dtype=torch.long)
        
        for batch_idx, indices in enumerate(indices_list):
            # Skip if indices is [-1] (no idiom)
            if indices == [-1]:
                continue
            
            # Mark the first index as B (class 0) and the rest as I (class 1)
            for i, idx in enumerate(indices):
                if idx < seq_length:  # Ensure index is within sequence length
                    if i == 0:
                        labels[batch_idx, idx] = 0  # B
                    else:
                        labels[batch_idx, idx] = 1  # I
        
        return labels 