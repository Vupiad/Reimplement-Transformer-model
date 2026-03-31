import os
import torch
from transformers import BertTokenizer

# Suppress the symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class TransformerTokenizer:
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        """
        Initializes the tokenizer using BERT's subword vocabulary.
        """
        self.max_length = max_length
        try:
            # Try loading from local cache first to speed up execution
            self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        except OSError:
            # Download from Hugging Face if not cached locally
            self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode(self, text):
        """
        Converts a raw string of text into token IDs and an attention mask.
        """
        # Call the tokenizer directly to perform all preprocessing steps
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,      # Adds [CLS] at the start and [SEP] at the end
            max_length=self.max_length,   # Standardizes sequence length
            padding='max_length',         # Pads shorter sentences with 0s
            truncation=True,              # Truncates longer sentences
            return_tensors='pt'           # Returns PyTorch tensors ready for the model
        )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }

    def decode(self, token_ids):
        """
        Utility function to convert IDs back to text for debugging.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

