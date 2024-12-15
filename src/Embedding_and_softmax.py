import torch
import torch.nn as nn

class PretrainedEmbedding(nn.Module):
    def __init__(self, embedding_matrix):
        """
        Args:
            embedding_matrix: Pretrained embedding matrix as a PyTorch tensor of shape (vocab_size, embed_dim).
        """
        super(PretrainedEmbedding, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding_dim = embed_dim
        # Define the embedding layer with pretrained weights
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)
        # Load pretrained weights
        self.embeddings.weight.data.copy_(src=embedding_matrix)

        # Optionally, freeze embeddings to prevent updates during training
        self.embeddings.weight.requires_grad = False  # Set to True if fine-tuning is needed

    def forward(self, word_indices):
        """
        Args:
            word_indices: Tensor of word indices (batch_size, sequence_length).

        Returns:
            Embedding vectors for the input words (batch_size, sequence_length, embed_dim).
        """
        return self.embeddings(word_indices) * torch.sqrt(input=torch.tensor(data=self.embed_dim, dtype=torch.float32))
