from Encoder import Encoder
from Decoder import Decoder
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()

        # Encoder and Decoder
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Source sequence (batch_size, src_seq_len)
            tgt: Target sequence (batch_size, tgt_seq_len)
            src_mask: Mask for source input (optional)
            tgt_mask: Mask for target input (optional)
        """
        # Pass through the encoder
        encoder_output = self.encoder(src, src_mask)
        
        # Pass through the decoder
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        return decoder_output