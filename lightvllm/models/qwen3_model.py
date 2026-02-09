import torch
from torch import nn
from lightvllm.layers.layernorm import RMSNorm
from lightvllm.layers.attention import Attention
from lightvllm.layers.activation import SiluAndMul
from lightvllm.layers.rotary_embedding import get_rope
from lightvllm.layers.embed_head import Embedding, LMHead
from lightvllm.layers.linear import Linear, QKVLinear, GateUpLinear

from lightvllm.models.qwen3_config import Qwen3Config


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_theta: float,
        max_position_embeddings: int,
        rms_norm_eps: float
        ):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.qkv_proj = QKVLinear(hidden_size, head_dim, num_attention_heads, num_key_value_heads)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.rotary_emb = get_rope(head_dim, max_position_embeddings, rope_theta)
        self.attn = Attention()
        self.o_proj = Linear(head_dim * num_attention_heads, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden_states)

        bsz, seq_len, _ = hidden_states.shape
        q = q.view(bsz, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, attention_mask)
        output = self.o_proj(attn_output.flatten(2, 3))
        return output

    
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_up_proj = GateUpLinear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        down = self.down_proj(self.act_fn(gate_up))
        return down


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, positions, attention_mask)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config
    ):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, positions, attention_mask)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", "gate"),
        "up_proj": ("gate_up_proj", "up"),
    }

    def __init__(
        self,
        config: Qwen3Config
    ):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = LMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, attention_mask)
        logits = self.lm_head(hidden_states[:, -1, :])
        return logits
