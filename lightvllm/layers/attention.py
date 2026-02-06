import torch
import torch.nn as nn
from flash_attn import flash_attn_func

class Attention(nn.Module):

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        num_q_heads = q.size(1)
        num_kv_heads = k.size(1)
        if num_q_heads != num_kv_heads:
            group_size = num_q_heads // num_kv_heads
            k = k.repeat_interleave(group_size, dim=1)
            v = v.repeat_interleave(group_size, dim=1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        
        seq_len = q.size(-2)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
        attn_weights = torch.softmax(scores, dim=-1)
        
        return torch.matmul(attn_weights, v)


class FlashAttention(nn.Module):

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 使用 flash_attn 实现 causal attention
        return flash_attn_func(q, k, v, causal=True)


class PagedAttention(nn.Module):

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        raise NotImplementedError