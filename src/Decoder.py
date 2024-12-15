import torch.nn as nn
from Position_wise_Feed_Forward_Networks import PositionWiseFeedForwardNN
from scaled_dot_product_attention import MultiHeadAttention
from Positional_encoding import PositionalEncoding
from Embedding_and_softmax import PretrainedEmbedding
import math 

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model)  # Masked self-attention
        self.enc_dec_attn = MultiHeadAttention(num_heads, d_model)  # Encoder-Decoder attention
        self.feedforward = PositionWiseFeedForwardNN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target sequence embeddings (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder outputs (batch_size, src_seq_len, d_model)
            src_mask: Mask for source input (optional)
            tgt_mask: Mask for target input (optional)
        """
        # Masked self-attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Encoder-Decoder attention
        enc_dec_attn_output = self.enc_dec_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(enc_dec_attn_output))

        # Feedforward network
        ff_output = self.feedforward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len=5000, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = PretrainedEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            tgt: Target sequence (batch_size, tgt_seq_len)
            encoder_output: Encoder outputs (batch_size, src_seq_len, d_model)
            src_mask: Mask for source input (optional)
            tgt_mask: Mask for target input (optional)
        """
        # Token embeddings + positional encoding
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary size
        output = self.output_projection(x)

        return output
