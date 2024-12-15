import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: Dimension of the model (embedding size).
            max_len: Maximum sequence length for which positional encodings are precomputed.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Precompute positional encodings for the maximum sequence length
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2)

        # Compute sine and cosine terms
        # Page 6 section 3.5 the formulas
        positional_encoding[:, 0::2] = torch.sin(position * div_term)  # Apply to even indices
        positional_encoding[:, 1::2] = torch.cos(position * div_term)  # Apply to odd indices

        # Add a batch dimension and register as a buffer
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            Tensor of shape (batch_size, sequence_length, d_model) with positional encodings added.
        """
        seq_len = x.size(1)
        # Add positional encoding to input embeddings
        x = x + self.positional_encoding[:, :seq_len, :]
        return x
