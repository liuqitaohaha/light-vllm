import torch
from lightvllm.layers.sampler import Sampler
from lightvllm.utils.loader import load_model
from lightvllm.sampling_params import SamplingParams
from lightvllm.models.qwen3_config import Qwen3Config
from lightvllm.models.qwen3_model import Qwen3ForCausalLM
from lightvllm.models.qwen3_moe_config import Qwen3MoeConfig
from lightvllm.models.qwen3_moe_model import Qwen3MoeForCausalLM


class LLM:
    
    def __init__(self, model: str):
        self.config = Qwen3MoeConfig(model+"/config.json")

        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)

        self.model = Qwen3MoeForCausalLM(self.config)
        load_model(self.model, path=model)
        
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.get_default_dtype())

        self.sampler = Sampler()
        print("LLM init successfully")

    @torch.inference_mode()
    def generate(
        self, 
        prompts: list[list[int]], 
        sampling_params: list[SamplingParams], 
        attention_mask: list[list[int]] = None
    ):
        orig_lens = [len(p) for p in prompts]
        temperatures = [sp.temperature for sp in sampling_params]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device="cuda")
        
        if attention_mask is None:
            attention_mask = [[1] * len(p) for p in prompts]
        
        done = [False] * len(prompts)

        step = 0
        while False in done:
            step += 1
            print(f"generate step {step}")

            active_indices = [i for i, f in enumerate(done) if not f]

            active_input_ids = [prompts[i] for i in active_indices]
            input_ids = torch.tensor(active_input_ids, dtype=torch.long, device="cuda")
            positions = torch.arange(0, input_ids.shape[1], dtype=torch.long, device="cuda")

            active_attention_mask = [attention_mask[i] for i in active_indices]
            active_attention_mask = torch.tensor(active_attention_mask, dtype=torch.long, device="cuda")

            logits = self.model(input_ids, positions, attention_mask=active_attention_mask)
            next_tokens = self.sampler(logits, temperatures[active_indices])

            for j, i in enumerate(active_indices):
                prompts[i].append(next_tokens[j].item())
                attention_mask[i].append(1)
                if (not sampling_params[i].ignore_eos and \
                    next_tokens[j].item() == self.config.eos_token_id) or \
                    len(prompts[i]) - orig_lens[i] == sampling_params[i].max_tokens:
                    done[i] = True

        generated_outputs = [p[l:] for p, l in zip(prompts, orig_lens)]
        return generated_outputs 
