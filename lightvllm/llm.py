from lightvllm.utils.loader import load_model
from lightvllm.sampling_params import SamplingParams
from lightvllm.models.qwen3_moe_config import Qwen3MoeConfig


class LLM:
    def __init__(self, model: str):
        self.config = Qwen3MoeConfig(model)

        torch.set_default_dtype(self.config.torch_dtype)
        torch.set_default_device("cuda")

        self.model = Qwen3MoeForCausalLM(self.config)
        load_model(self.model, path=model)
        
        torch.set_default_dtype(torch.get_default_dtype())
        torch.set_default_device("cpu")

        self.sampler = Sampler()

    def generate(self, prompts: list[list[int]], sampling_params: list[SamplingParams]):
        temperatures = [sp.temperature for sp in sampling_params]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device="cuda")

        done = [False] * len(prompts)

        while False in done:
            active_indices = [i for i, f in enumerate(done) if not f]

            active_input_ids = [prompts[i] for i in active_indices]
            input_ids = torch.tensor(active_input_ids, dtype=torch.long, device="cuda")
            positions = torch.arange(0, input_ids.shape[1], dtype=torch.long, device="cuda")

            logits = self.model(positions, input_ids)
            next_tokens = self.sampler(logits, temperatures[active_indices])

            for j, i in enumerate(active_indices):
                if next_tokens[j].item() == self.config.eos_token_id or \
                    len(prompts[i]) >= sampling_params[i].max_tokens:
                    done[i] = True
                else:
                    prompts[i].append(next_tokens[j].item())

        return prompts
