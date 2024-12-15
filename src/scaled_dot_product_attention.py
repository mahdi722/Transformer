import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        """
        Initialize Single Head Attention module
        
        Args:
            num_heads (int): Number of attention heads 
            d_model (int): Total dimension of the model's embedding space
        
        Attributes:
            d_k (int): Dimension of each attention head 
                       Calculated as: d_model // num_heads
            d_model (int): Total model dimension 
            num_heads (int): Number of attention heads
        
        Dimension Calculations:
        - If d_model = 512 and num_heads = 8
        - d_k will be 512 // 8 = 64
        """
        super(SingleHeadAttention, self).__init__()
        """ Paper Reference:
                Page 3, Figure 2 (Architecture Diagram)
                Equation (1): Splitting d_model into h heads"""
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads

        # Linear projections for Query, Key, Value
        # Input: (batch_size, sequence_length, d_model)
        # Output: (batch_size, sequence_length, d_model)
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        """Paper Reference:
            Page 3, Section 3.2.2
            "In this work we use h = 8 parallel attention layers, or heads"
            Implements the linear projections for queries, keys, and values"""
    def forward(self, query, key, value, mask=None):
        """
        Compute single head attention
        
        Args:
            query (Tensor): Query tensor 
                - Shape: (batch_size, sequence_length, d_model)
            key (Tensor): Key tensor 
                - Shape: (batch_size, sequence_length, d_model)
            value (Tensor): Value tensor 
                - Shape: (batch_size, sequence_length, d_model)
            mask (Tensor, optional): Attention mask 
                - Shape: (batch_size, 1, sequence_length, sequence_length)
        
        Returns:
            output (Tensor): Attention output 
                - Shape: (batch_size, sequence_length, d_model)
            attention_weight (Tensor): Attention weights 
                - Shape: (batch_size, num_heads, sequence_length, sequence_length)
        """
        batch_size, seq_len, _ = query.size()

        # Project and split into heads
        # Q, K, V shape transformation:
        # (batch_size, sequence_length, d_model) 
        # -> (batch_size, sequence_length, num_heads, d_k) with view operation
        # -> (batch_size, num_heads, sequence_length, d_k) with transpose operation
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(dim0=2, dim1=1)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(dim0=2, dim1=1)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(dim0=2, dim1=1)
        # Compute attention scores
        # Q shape: (batch_size, num_heads, sequence_length, d_k)
        # K transposed shape: (batch_size, num_heads, d_k, sequence_length)
        # Result score shape: (batch_size, num_heads, sequence_length, sequence_length)
        scores = torch.matmul(input=Q, other=K.transpose(dim0=-2, dim1=-1)) / torch.sqrt(input=torch.tensor(data=self.d_k, dtype=torch.float32))
        """Paper Reference:

            Page 3, Equation (2): Scaled Dot-Product Attention
            Attention(Q, K, V) = softmax(QK^T / √d_k)V
            The √d_k scaling prevents dot products from growing large in magnitude"""
        # Apply mask if provided
        # Masks out certain attention connections by setting them to -inf
        if mask is not None:
            scores = scores.masked_fill(mask == 0, value=torch.float('-inf'))
        """Paper Reference:

            Page 6, Section 3.3 "Masking"
            Used in encoder-decoder attention and self-attention layers
            Prevents attending to padding tokens in the input sequence"""
        # Compute attention weights using softmax
        # Softmax applied across the last dimension
        # attention_weight shape: (batch_size, num_heads, sequence_length, sequence_length)
        attention_weight = F.softmax(input=scores, dim=-1)
        """Paper Reference:

            Page 3, Equation (2): Completing the Scaled Dot-Product Attention
            Softmax normalizes the attention scores
            Matrix multiplication with values computes the weighted representation"""
        # Compute attention output
        # Multiply attention weights with value tensor
        # Output shape: (batch_size, num_heads, sequence_length, d_k)
        output = torch.matmul(input=attention_weight, other=V)

        # Reshape output back to original dimensions
        # (batch_size, num_heads, sequence_length, d_k) 
        # -> (batch_size, sequence_length, num_heads, d_k) with transpose operation
        # -> (batch_size, sequence_length, d_model) with view operation
        output = output.transpose(dim0=1, dim1=2).contiguous().view(batch_size, seq_len, self.d_model)

        return output, attention_weight

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        """
        Initialize Multi-Head Attention module
        
        Args:
            num_heads (int): Number of attention heads
            d_model (int): Total dimension of the model's embedding space
        
        Attributes:
            num_heads (int): Number of attention heads
            d_model (int): Total model dimension
            d_k (int): Dimension of each attention head
                       Calculated as: d_model // num_heads
            W_o (nn.Linear): Final linear projection layer
        
        Dimension Calculations:
        - If d_model = 512 and num_heads = 8
        - d_k will be 512 // 8 = 64
        """
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # Final linear projection layer
        # Input: (batch_size, sequence_length, d_model)
        # Output: (batch_size, sequence_length, d_model)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        """
        Paper Reference:

            Page 3, Last paragraph of Section 3.2
            "Finally, the multi-head attention performs a linear projection"
            Concatenates head outputs and projects to d_model dimension
        """
        # Single instance of SingleHeadAttention for efficient computation
        self.single_head_attention = SingleHeadAttention(num_heads=num_heads, d_model=d_model)

    def forward(self, query, key, value, mask=None):
        """
        Compute multi-head attention
        
        Args:
            query (Tensor): Query tensor 
                - Shape: (batch_size, sequence_length, d_model)
            key (Tensor): Key tensor 
                - Shape: (batch_size, sequence_length, d_model)
            value (Tensor): Value tensor 
                - Shape: (batch_size, sequence_length, d_model)
            mask (Tensor, optional): Attention mask 
                - Shape: (batch_size, 1, sequence_length, sequence_length)
        
        Returns:
            output (Tensor): Multi-head attention output 
                - Shape: (batch_size, sequence_length, d_model)
        """

        # Perform single-pass multi-head attention
        # Input and output shapes: (batch_size, sequence_length, d_model)
        output, _ = self.single_head_attention(query, key, value, mask)
        
        # Final linear projection
        # Applies linear transformation to the attention output
        # Helps in mixing information from different heads
        output = self.W_o(output) # (batch_size, sequence_length, d_model)

        return output
    
# batch_size = 4
# sequence_length = 6
# d_model = 512
# num_heads = 8
# query = torch.randn(batch_size, sequence_length, d_model)  # Example input tensor
# key = torch.randn(batch_size, sequence_length, d_model)    # Example key tensor
# value = torch.randn(batch_size, sequence_length, d_model)  # Example value tensor

# # Instantiate and use the MultiHeadAttention
# multi_head_attention = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
# output = multi_head_attention(query=query, key=key, value=value)
# print(output.shape)  # Should be (batch_size, sequence_length, d_model)