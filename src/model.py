import torch
import torch.nn as nn
import copy
from src.attention import MultiHeadAttention
from src.embeddings import TransformerEmbedding
from src.layers import (
    Encoder, Decoder, EncoderDecoder, Generator, 
    EncoderLayer, DecoderLayer, PositionwiseFeedForward
)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Construct a complete Transformer (Encoder-Decoder) model from hyperparameters.
    
    Args:
        src_vocab: Size of source vocabulary
        tgt_vocab: Size of target vocabulary
        N: Number of individual layers in Encoder and Decoder
        d_model: Dimensionality of the model's hidden states
        d_ff: Dimensionality of the Feed-Forward network's inner layer
        h: Number of attention heads
        dropout: Dropout rate
    """
    c = copy.deepcopy
    
    # 1. Initialize core attention and feed-forward modules
    attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    
    # 2. Build the Transformer model
    model = EncoderDecoder(
        # Encoder: stack of N EncoderLayers
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        
        # Decoder: stack of N DecoderLayers
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        
        # Source Embedding
        TransformerEmbedding(src_vocab, d_model, dropout=dropout),
        
        # Target Embedding
        TransformerEmbedding(tgt_vocab, d_model, dropout=dropout),
        
        # Final output generator
        Generator(d_model, tgt_vocab)
    )
    
    # 3. Initialize parameters (Xavier initialization is standard for Transformers)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model

if __name__ == "__main__":
    # Example: Create a model for French-to-English translation
    # Assuming vocab sizes of 10000 for both
    temp_model = make_model(src_vocab=10000, tgt_vocab=10000)
    print(f"Model initialized with {sum(p.numel() for p in temp_model.parameters()):,} parameters.")
