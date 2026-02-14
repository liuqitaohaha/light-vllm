from transformers import AutoTokenizer
from lightvllm import LLM, SamplingParams


def main():
    # model_path = "/mnt/liuqitao-default/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    model_path = "/mnt/liuqitao-default/.cache/huggingface/hub/models--TroyDoesAI--Qwen3-15B-A2B-Base/snapshots/d0e6f9c8e7fcd17d3734fb6c60ed0439f7b138d1"
    
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    inputs = [
        "用中文介绍一下你自己",
        "如何看待中美两国在AI领域的竞争和合作",
    ]
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": input}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for input in inputs
    ]
    inputs_tokenzied = tokenizer(inputs, padding=True, add_special_tokens=True)
    prompts = inputs_tokenzied.input_ids
    attention_mask = inputs_tokenzied.attention_mask
    sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=128),
        SamplingParams(temperature=0.9, max_tokens=64)
    ]

    completions = llm.generate(prompts, sampling_params, attention_mask=attention_mask)
    
    outputs = tokenizer.batch_decode(completions, skip_special_tokens=True)
    for input, output in zip(inputs, outputs):
        print(f"Input: {input!r}")
        print(f"Output: {output!r}")


if __name__ == "__main__":
    main()
