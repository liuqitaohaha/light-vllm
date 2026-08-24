import os
import json
from dataclasses import dataclass


@dataclass
class Qwen3MoeConfig:

    head_dim: int = 128
    hidden_size: int = 1024
    max_position_embeddings: int = 40960
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_theta: int = 1000000
    tie_word_embeddings: bool = True
    vocab_size: int = 151936
    eos_token_id: int = 151645
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768

    def __init__(self, json_path: str):
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"File Not Found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        for key, value in cfg.items():
            setattr(self, key, value)