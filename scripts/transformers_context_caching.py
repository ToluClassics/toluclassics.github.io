import copy
import inspect
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None
prepare_inputs_params = set(inspect.signature(model.prepare_inputs_for_generation).parameters)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def build_user_prefix_cache(user_prefix: str):
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prefix}],
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    inputs = tokenizer(
        prefix_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    return outputs.past_key_values, inputs["input_ids"].shape[1], prefix_text


def generate_with_prefix_cache(user_text, prefix_cache, prefix_len, prefix_text, user_prefix):
    kv_cache = copy.deepcopy(prefix_cache)

    full_prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prefix + user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    suffix_text = full_prompt_text[len(prefix_text):]

    inputs = tokenizer(
        suffix_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    suffix_len = inputs["input_ids"].shape[1]

    full_attention_mask = torch.ones(
        (1, prefix_len + suffix_len),
        device=model.device,
        dtype=torch.long,
    )

    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": full_attention_mask,
        "past_key_values": kv_cache,
        "max_new_tokens": 128,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    if "cache_position" in prepare_inputs_params:
        generate_kwargs["cache_position"] = torch.arange(
            prefix_len,
            prefix_len + suffix_len,
            device=model.device,
        )

    with torch.no_grad():
        out = model.generate(**generate_kwargs)

    generated = out.sequences[0, suffix_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_without_cache(user_text, user_prefix):
    full_prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prefix + user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        full_prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    generated = out.sequences[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def timed_call(fn, *args):
    start = time.perf_counter()
    output = fn(*args)
    elapsed = time.perf_counter() - start
    return output, elapsed

# Common prefix.
prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: "
)

# Sample prompts.
prompts = [
    "Who do i teach?",
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]*100

uncached_generation_total = 0.0
for prompt in prompts:
    uncached_output, uncached_time = timed_call(generate_without_cache, prompt, prefix)
    uncached_generation_total += uncached_time

cache, cache_build_time = timed_call(build_user_prefix_cache, prefix)
cached_generation_total = 0.0


for prompt in prompts:
    cached_output, cached_time = timed_call(
        generate_with_prefix_cache, prompt, cache[0], cache[1], cache[2], prefix
    )
    cached_generation_total += cached_time



print(f"Cache build time: {cache_build_time:.4f}s")
print(f"Total with cache: {cached_generation_total:.4f}s")
print(f"Total without cache: {uncached_generation_total:.4f}s")
