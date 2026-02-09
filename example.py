from transformers import AutoTokenizer
from lightvllm import LLM, SamplingParams


def main():
    model_path = "/mnt/liuqitao-default/modelzoo/Qwen3-8B/"
    
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    inputs = [
        "介绍一下你自己,简短点,不超过30字",
        "列出所有100以内的质数",
    ]
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": input}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for input in inputs
    ]
    inputs = tokenizer(inputs, padding=True, add_special_tokens=True)
    prompts = inputs.input_ids
    attention_mask = inputs.attention_mask
    sampling_params = SamplingParams(temperature=0.9, max_tokens=256)

    completions = llm.generate(prompts, [sampling_params] * len(prompts), attention_mask=attention_mask)
    
    outputs = tokenizer.batch_decode(completions, skip_special_tokens=True)
    for input, output in zip(inputs, outputs):
        print(f"Input: {input!r}")
        print(f"Output: {output!r}")


if __name__ == "__main__":
    main()
