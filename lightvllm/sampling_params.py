from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 256
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.max_tokens > 0, "max_tokens must be greater than 0"
