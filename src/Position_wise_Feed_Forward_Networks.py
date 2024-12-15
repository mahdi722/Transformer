import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForwardNN(nn.Module):
    """
    Paper reference: page 5, section 3.3
    """
    def __init__(self, d_model, dff, dropout=0.2):
        super(PositionWiseFeedForwardNN, self).__init__()
        self.FNN = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=dff, bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=dff, out_features=d_model, bias=True)
        )

    def forward(self, x):
        return self.FNN(x)
