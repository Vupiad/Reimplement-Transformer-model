import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding to inject word order 
    information into the model.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create a matrix of shape (max_len, d_model) full of zeros
        pe = torch.zeros(max_len, d_model)
        
        # Create a column vector of positions: [[0], [1], [2], ..., [max_len-1]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the denominator term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (2i) [cite: 167-168, 173]
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (2i+1) [cite: 167-168, 174]
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: shape becomes (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as a buffer so PyTorch knows this is fixed data, 
        # not a weight that needs to be updated during backpropagation.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds the positional encoding to the token embeddings.
        """
        # x shape: (batch_size, sequence_length, d_model)
        # We slice self.pe to match the actual sequence length of x [cite: 111-115]
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerEmbedding(nn.Module):
    """
    Combines token embeddings and positional encodings into a single module.
    """
    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        # The embedding matrix E in R^(V x d_model) 
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # The positional encoding matrix P [cite: 115]
        self.pos_emb = PositionalEncoding(d_model, max_len)
        
        # Dropout is commonly added here to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        # 1. Convert integer token IDs to dense vectors (X_tok) [cite: 77-80]
        x = self.token_emb(input_ids)
        
        # 2. Add sinusoidal positional information (X = X_tok + P) 
        x = self.pos_emb(x)
        
        return self.dropout(x)


