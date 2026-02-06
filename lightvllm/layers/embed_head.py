import torch
from torch import nn
from torch.nn import functional as F


class Embedding(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, embed_dim))
    
    def forward(self, x: torch.Tensor):
        y = F.embedding(x, self.weight)
        return y


class LMHead(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, embed_dim))

    def forward(self, x: torch.Tensor):
        y = F.linear(x, self.weight)
        return y