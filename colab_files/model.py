import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from typing import Optional, Tuple, List
from torchcrf import CRF  # This will be installed in Colab via pip install pytorch-crf

class MLP(nn.Module):
    """
    Multi-layer Perceptron for classification
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class IdiomDetectionModel(nn.Module):
    """
    Idiom Detection Model using XLM-RoBERTa-large backbone with BiLSTM, MLP and CRF layers.
    This model predicts whether each token is beginning of an idiom (B),
    inside an idiom (I), or outside an idiom (O).
    """
    def __init__(self, model_name, dropout_rate=0.1, use_crf=True, use_bilstm=True, 
                 use_mlp=True, lstm_hidden_dim=768, mlp_hidden_dim=512, class_weights=None):
        """
        Initialize the idiom detection model.
        
        Args:
            model_name: Name of the pretrained model to use as the encoder
            dropout_rate: Dropout rate for the classification head
            use_crf: Whether to use a CRF layer for sequence labeling
            use_bilstm: Whether to use a BiLSTM layer after the transformer encoder
            use_mlp: Whether to use a MLP classifier instead of a simple linear layer
            lstm_hidden_dim: Hidden dimension of the LSTM layer (per direction)
            mlp_hidden_dim: Hidden dimension of the MLP classifier
            class_weights: Optional weights for the loss function to handle class imbalance
        """
        super().__init__()
        
        # Load model configuration
        self.config = XLMRobertaConfig.from_pretrained(model_name)
        
        # Load pre-trained encoder
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        
        # BiLSTM layer for sequence modeling
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.lstm = nn.LSTM(
                self.config.hidden_size, 
                lstm_hidden_dim, 
                num_layers=1, 
                bidirectional=True,
                batch_first=True,
                dropout=0.0  # No dropout between LSTM layers since we only have 1 layer
            )
            # Output dimension of BiLSTM is 2 * hidden_dim due to bidirectionality
            self.feature_dim = lstm_hidden_dim * 2
        else:
            self.feature_dim = self.config.hidden_size
        
        # Dropout before classification
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head for BIO tagging (3 classes)
        self.use_mlp = use_mlp
        if use_mlp:
            self.classifier = MLP(
                input_dim=self.feature_dim,
                hidden_dim=mlp_hidden_dim,
                output_dim=3,  # 3 classes: B(0), I(1), O(2)
                dropout_rate=dropout_rate
            )
        else:
            self.classifier = nn.Linear(self.feature_dim, 3)  # 3 classes: B(0), I(1), O(2)
        
        # CRF layer for sequence labeling
        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(3, batch_first=True)  # 3 classes: B(0), I(1), O(2)
        
        # Store class weights for weighted loss
        self.class_weights = class_weights
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor indicating which tokens are padding
            labels: Optional tensor of token labels (0=B, 1=I, 2=O)
            
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
        
        # Pass through BiLSTM if enabled
        if self.use_bilstm:
            # Pack padded sequence for efficient computation
            packed_sequence = nn.utils.rnn.pack_padded_sequence(
                sequence_output, 
                attention_mask.sum(1).cpu(),  # Calculate actual lengths
                batch_first=True,
                enforce_sorted=False
            )
            
            # Apply BiLSTM
            packed_lstm_output, _ = self.lstm(packed_sequence)
            
            # Unpack the sequence
            sequence_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_lstm_output,
                batch_first=True,
                padding_value=0.0,
                total_length=sequence_output.size(1)
            )
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Apply classification layer (MLP or Linear)
        logits = self.classifier(sequence_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if self.use_crf:
                # Create a mask to ignore padding tokens (and special tokens marked as -100)
                # CRF requires the first position in each sequence to be valid
                # Create mask where all first positions are 1 (required by CRF)
                batch_size, seq_length = labels.shape
                mask = torch.zeros_like(attention_mask, dtype=torch.bool)
                
                # Set first position to True for each sequence
                mask[:, 0] = True  
                
                # Set other positions according to attention mask and valid labels
                for i in range(1, seq_length):
                    mask[:, i] = (labels[:, i] != -100) & (attention_mask[:, i] == 1)
                
                # Convert labels from -100 to valid indices for CRF (we'll use 2/O for these tokens)
                crf_labels = labels.clone()
                # First position should be a valid tag (use O/2 for CLS token)
                crf_labels[:, 0] = 2
                # Other -100 positions to 2 (O tag)
                crf_labels[crf_labels == -100] = 2
                
                # CRF calculates negative log likelihood
                loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
            else:
                # Use weighted CrossEntropyLoss if class_weights is provided
                weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
                
                # Reshape for CrossEntropyLoss which expects [N, C] for logits and [N] for labels
                # where N is the number of active tokens
                active_loss_mask = labels.view(-1) != -100
                active_logits = logits.view(-1, 3)[active_loss_mask]
                active_labels = labels.view(-1)[active_loss_mask]
                
                if active_logits.shape[0] > 0:  # Ensure there are active tokens to calculate loss for
                    loss = loss_fn(active_logits, active_labels)
                else:
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return logits, loss
    
    def convert_logits_to_indices(self, logits, attention_mask):
        """
        Convert logits to idiom indices using CRF for prediction if enabled.
        
        Args:
            logits: Tensor of logits with shape [batch_size, seq_len, 3]
            attention_mask: Tensor indicating which tokens are padding (1 for real token, 0 for padding)
            
        Returns:
            List of lists, where each inner list contains the subword indices of tokens
            predicted as part of an idiom (B or I) for the corresponding example, or [-1] if no idiom found
        """
        batch_size = logits.size(0)
        batch_idiom_indices = []
        
        if self.use_crf:
            # CRF requires the first position in each sequence to be valid
            batch_size, seq_length = attention_mask.shape
            mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            
            # Set first position to True for each sequence
            mask[:, 0] = True
            
            # Set other positions according to attention mask
            for i in range(1, seq_length):
                mask[:, i] = attention_mask[:, i] == 1
            
            # Use CRF to decode the best tag sequence
            predictions = self.crf.decode(logits, mask=mask)
            
            # For each example in the batch
            for i in range(batch_size):
                # Get the predicted tags for this example
                example_predictions = predictions[i]
                
                # Get indices of active tokens for this example (excluding the first CLS token)
                active_tokens_indices = torch.where(attention_mask[i] == 1)[0]
                
                if len(active_tokens_indices) <= 1:  # Only CLS token, no real content
                    batch_idiom_indices.append([-1])
                    continue
                
                # Find positions with B (0) or I (1) tags in the example predictions
                # Skip the first token (CLS token) when extracting idiom indices
                idiom_indices = []
                for j, tag in enumerate(example_predictions):
                    if j > 0 and j < len(active_tokens_indices) and (tag == 0 or tag == 1):  # B or I tag, skip first
                        idiom_indices.append(active_tokens_indices[j].item())
                
                if not idiom_indices:
                    batch_idiom_indices.append([-1])
                else:
                    batch_idiom_indices.append(idiom_indices)
        else:
            # Use argmax for prediction if CRF is not enabled (previous behavior)
            predictions = torch.argmax(logits, dim=-1)  # Shape: [batch_size, seq_len], values 0,1,2
            
            for i in range(batch_size):
                # Get indices of active (non-padding) tokens for the current example
                active_tokens_indices = torch.where(attention_mask[i] == 1)[0]
                
                if len(active_tokens_indices) == 0:
                    batch_idiom_indices.append([-1])
                    continue

                # Get predictions only for these active tokens
                active_predictions = predictions[i][active_tokens_indices]
                
                # Find where the prediction is B (0) or I (1) among the active tokens
                is_B_or_I_mask = (active_predictions == 0) | (active_predictions == 1)
                idiom_relative_indices = torch.where(is_B_or_I_mask)[0]
                
                # Map these relative indices back to their original positions in the sequence
                idiom_original_subword_indices = active_tokens_indices[idiom_relative_indices].tolist()
                
                if len(idiom_original_subword_indices) == 0:
                    batch_idiom_indices.append([-1])
                else:
                    batch_idiom_indices.append(idiom_original_subword_indices)
        
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
            
            # Mark the indices as B (first token) and I (remaining tokens)
            if len(indices) > 0:
                # Process the indices to find B and I tags based on continuity
                for i, idx in enumerate(sorted(indices)):
                    if idx < seq_length:
                        # If this is the first index or the previous index is not adjacent
                        if i == 0 or (sorted(indices)[i-1] != idx-1):
                            labels[batch_idx, idx] = 0  # B
                        else:
                            labels[batch_idx, idx] = 1  # I
        
        return labels 