import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.weight_loader = self.weight_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)


class QKVLinear(Linear):

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
    ):
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        output_size = (num_attention_heads + num_key_value_heads * 2) * head_dim
        super().__init__(hidden_size, output_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str):
        assert shard_id in ["q", "k", "v"]
        if shard_id == "q":
            shard_size = self.head_dim * self.num_attention_heads
            shard_offset = 0
        elif shard_id == "k":
            shard_size = self.head_dim * self.num_key_value_heads
            shard_offset = self.head_dim * self.num_attention_heads
        elif shard_id == "v":
            shard_size = self.head_dim * self.num_key_value_heads
            shard_offset = self.head_dim * (self.num_attention_heads + self.num_key_value_heads)
        
        param.data[shard_offset:shard_offset+shard_size, :] = loaded_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = super().forward(x)
        q_size = self.head_dim * self.num_attention_heads
        kv_size = self.head_dim * self.num_key_value_heads
        return out.split([q_size, kv_size, kv_size], dim=-1)


class GateUpLinear(Linear):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int
    ):
        self.intermediate_size = intermediate_size
        output_size = intermediate_size * 2
        super().__init__(hidden_size, output_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str):
        assert shard_id in ["gate", "up"]
        if shard_id == "gate":
            shard_size = self.intermediate_size
            shard_offset = 0
        elif shard_id == "up":
            shard_size = self.intermediate_size
            shard_offset = self.intermediate_size
        
        param.data[shard_offset:shard_offset+shard_size, :] = loaded_weight
