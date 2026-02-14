import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_mask = temperatures.eq(0).unsqueeze(dim=1)
        logits = logits.float().div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)

        greedy_tokens = probs.argmax(dim=-1)
        random_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        
        sample_tokens = torch.where(greedy_mask.squeeze(dim=1), greedy_tokens, random_tokens)
        return sample_tokens
