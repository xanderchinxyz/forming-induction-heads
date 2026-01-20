import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Combined projection for Q, K, V
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Store attention weights for visualization
        self.attn_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.W_qkv(x)  # (batch, seq, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (batch, seq, d_model)
        
        # Reshape to (batch, n_heads, seq, d_head)
        # Could just use one transpose instead of two for K, but having two makes it more readable
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Causal mask: prevent attending to future tokens
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Store for visualization (detach to avoid memory issues)
        self.attn_weights = attn_weights.detach()
        
        # Apply attention to values
        out = attn_weights @ v  # (batch, n_heads, seq, d_head)
        
        # Reshape back to (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(out)


def create_sinusoidal_embeddings(max_context_len: int, d_model: int) -> torch.Tensor:
    """
    Creates fixed sinusoidal positional embeddings.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    position = torch.arange(max_context_len).unsqueeze(1)  # (max_context_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d_model/2,)
    
    pe = torch.zeros(max_context_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
    
    return pe  # (max_context_len, d_model)

class AttentionOnlyTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        n_layers: int, 
        n_heads: int,
        max_context_len: int,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        
        # Token embedding (learned)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding (fixed sinusoidal - not learned)
        pe = create_sinusoidal_embeddings(max_context_len, d_model)
        self.register_buffer('position_embedding', pe)
        
        # Stack of attention blocks (no MLP/FFN)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # Output projection to vocab
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        
        # Get embeddings (position_embedding is already on correct device via register_buffer)
        x = self.token_embedding(tokens) + self.position_embedding[:seq_len]
        
        # Pass through attention blocks with residual connections
        for attn_block in self.attention_blocks:
            x = x + attn_block(x)
        
        # Project to vocabulary, shape: [b, s, vocab_size]
        logits = self.unembed(x)
        
        return logits
        
