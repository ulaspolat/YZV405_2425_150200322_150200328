import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast
from transformers import XLMRobertaTokenizer
from typing import List, Dict, Tuple, Optional


class IdiomDetectionDataset(Dataset):
    """
    Dataset for idiom detection task using tokenized sentences and idiom indices.
    """
    def __init__(self, file_path, tokenizer, max_length=128, language_filter=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load and optionally filter the dataset
        self.data = pd.read_csv(file_path)
        if language_filter:
            self.data = self.data[self.data['language'] == language_filter]

        self.has_labels = 'indices' in self.data.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        text = example['sentence']
        language = example['language']
        example_id = example['id']

        # Parse the word-level tokens and idiom indices
        word_tokens = ast.literal_eval(example['tokenized_sentence'])
        idiom_indices = self._parse_idiom_indices(example['indices']) if self.has_labels else []

        # Initialize sequences
        subword_ids = [self.tokenizer.cls_token_id]
        attention_mask = [1]
        labels = [0.0] if self.has_labels else None

        # Align subwords with idiom word indices
        for word_idx, word in enumerate(word_tokens):
            word_pieces = self.tokenizer.tokenize(word)
            word_piece_ids = self.tokenizer.convert_tokens_to_ids(word_pieces)

            subword_ids.extend(word_piece_ids)
            attention_mask.extend([1] * len(word_piece_ids))

            if self.has_labels:
                label = 1.0 if word_idx in idiom_indices else 0.0
                labels.extend([label] * len(word_piece_ids))

        subword_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)
        if self.has_labels:
            labels.append(0.0)

        # Padding
        pad_len = self.max_length - len(subword_ids)
        if pad_len > 0:
            subword_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            if self.has_labels:
                labels += [0.0] * pad_len
        else:
            subword_ids = subword_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            if self.has_labels:
                labels = labels[:self.max_length]

        # Final output
        result = {
            'id': example_id,
            'input_ids': torch.tensor(subword_ids),
            'attention_mask': torch.tensor(attention_mask),
            'language': language,
            'text': text
        }
        if self.has_labels:
            result['labels'] = torch.tensor(labels)

        return result

    def _parse_idiom_indices(self, label_str):
        if isinstance(label_str, float) and pd.isna(label_str):
            return []
        label_str = label_str.strip()
        if label_str in ('[]', '', '-1'):
            return []
        return ast.literal_eval(label_str)

    def collate_fn(self, batch):
        keys = ['id', 'input_ids', 'attention_mask', 'language', 'text']
        if self.has_labels:
            keys.append('labels')

        batch_dict = {k: [] for k in keys}
        for example in batch:
            for k in keys:
                batch_dict[k].append(example[k])

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
    language_filter: Optional[str] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
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

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        worker_init_fn=lambda worker_id: torch.manual_seed(seed + worker_id),
        collate_fn=train_dataset.collate_fn
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn
    )

    return train_dataloader, eval_dataloader
