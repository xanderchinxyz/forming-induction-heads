import torch

def generate_rrt(
    num_samples: int, 
    length: int, 
    vocab_size: int,
    seq_len: int | tuple[int, int],
    custom_seq: list[int] | None = None,
):
    """
    Generates a batch of Repeating Random Tokens.
    
    Args:
        num_samples: Number of sequences to generate (ignored if custom_prefix is provided)
        length: Total length of the output sequence
        vocab_size: Size of vocabulary for random tokens
        seq_len: Either a fixed int or a tuple (min, max) for a random sequence length
        custom_seq: Optional list of token IDs to use as the sequence (e.g., [1, 2, 3, 4])
                       If provided, this prefix will be repeated instead of random tokens.
    
    Returns:
        (prefix_len, data): Tuple of prefix length and data tensor of shape [num_samples, seq_len]
    """
    if custom_seq is not None:
        # Use custom prefix
        seq = torch.tensor(custom_seq).unsqueeze(0)
        seq_len = len(custom_seq)
        num_samples = 1  # Custom prefix always produces 1 sample
    else:
        # Determine prefix length (fixed or random)
        if isinstance(seq_len, tuple):
            seq_len = torch.randint(seq_len[0], seq_len[1], (1,)).item()
        
        # Generate random prefix
        seq = torch.randint(1, vocab_size, (num_samples, seq_len))
    
    # Repeat the prefix until we hit length
    repeats = (length // seq_len) + 1
    data = seq.repeat(1, repeats)[:, :length]
    
    return seq_len, data