import torch
import torch.nn as nn
from Position_wise_Feed_Forward_Networks import PositionWiseFeedForwardNN
from scaled_dot_product_attention import MultiHeadAttention
from Positional_encoding import PositionalEncoding
from Embedding_and_softmax import PretrainedEmbedding
import math 

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model)
        self.feedforward = PositionWiseFeedForwardNN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention and residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward network and residual connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len=5000, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = PretrainedEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Token embeddings + positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)

        return x