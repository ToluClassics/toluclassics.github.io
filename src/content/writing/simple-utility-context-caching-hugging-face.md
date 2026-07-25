---
title: "A Simple Utility for Context Caching with Hugging Face"
description: "Build a reusable prefix cache for independent Transformers requests, verify token boundaries, and check cached output against an uncached baseline."
publishedAt: 2026-03-25
updatedAt: 2026-07-25
draft: false
topics: ["Transformers", "Inference"]
featured: true
readingMinutes: 12
---

I wrote this utility because I needed something smaller than a serving stack. I wanted to load a model directly from Hugging Face, process one fixed piece of context once, and reuse it across several independent requests. I did not need vLLM, batching, scheduling, or a separate inference server. I just needed the cache.

I could not find a resource that matched that small use case closely enough, so I worked through the cache API and wrote the example I had been looking for.

The basic operation is short: run the fixed prefix through the model once, keep its key and value states, and reuse them. The difficult part is deciding exactly which tokens belong to that prefix and preventing one request from changing the cache used by the next one.

The utility is built around two invariants:

1. the cached tokens must be an exact prefix of every full request; and
2. every request must generate from an isolated copy of the prefilled cache.

It also compares greedy cached and uncached output before reporting any timing. Hugging Face Transformers documents this general technique as **prefix caching**; the code here narrows it to a direct-Transformers workflow with an explicit token boundary.

## What a KV cache contains

During autoregressive generation, each attention layer stores keys and values for tokens it has already processed. For a request with a reusable context, the sequence is:

```text
fixed context tokens + request tokens + generated tokens
```

The cache does not know that one region is a policy and another is a user request. It only represents tokens that precede the model’s next input. If we reuse the cache returned by a completed generation, it can contain the prior request and generated response as well as the fixed context.

For independent requests, we instead cache only the fixed context:

```text
                    ┌─ request A tokens → response A
fixed context cache ┤
                    └─ request B tokens → response B
```

Each branch starts from a deep copy. The preserved cache is never passed directly to `generate()`.

## Use a message boundary, then verify the tokens

In the example, the reusable context is a `system` message and each independent request is a `user` message:

```python
def _messages(self, prompt: str | None = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": self.context}]
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})
    return messages
```

This gives Qwen2’s chat template a natural boundary between the fixed and variable content. It is not safe to assume that the same boundary works for every tokenizer. Chat templates use model-specific control tokens, and some models do not support a `system` role at all.

The utility therefore tokenizes the context by itself, tokenizes the full request, and refuses to use the cache unless the first token IDs match exactly:

```python
def require_exact_prefix(prefix_ids, full_ids):
    prefix_length = len(prefix_ids)
    if prefix_length == 0:
        raise ValueError("The rendered reusable prefix contains no tokens.")
    if prefix_length > len(full_ids) or list(full_ids[:prefix_length]) != list(prefix_ids):
        raise ValueError(
            "The rendered reusable context is not an exact token prefix of the "
            "full request. This chat template cannot use the selected boundary."
        )
    return prefix_length
```

This token comparison matters even when one rendered string appears to begin with another. Tokenizers can choose a different token at the join between two text fragments. Character slicing cannot prove that a cache built from one tokenization is valid for another.

I also call `apply_chat_template(..., tokenize=True)` directly. Hugging Face recommends this over rendering text and tokenizing it in a second step because a second tokenizer call can accidentally duplicate special tokens.

## Build the reusable cache

The fixed message is rendered without an assistant generation prompt and passed through the model once:

```python
prefix_inputs = tokenizer.apply_chat_template(
    [{"role": "system", "content": context}],
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.inference_mode():
    prefix_cache = model(
        **prefix_inputs,
        use_cache=True,
    ).past_key_values
```

`model.eval()` is still required; `torch.inference_mode()` disables autograd bookkeeping but does not put the model in evaluation mode.

For a full request, the utility includes both messages and adds the assistant generation prompt. It verifies the prefix token IDs before calling the model:

```python
full_inputs = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": context},
        {"role": "user", "content": prompt},
    ],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

require_exact_prefix(
    prefix_inputs["input_ids"][0].tolist(),
    full_inputs["input_ids"][0].tolist(),
)
```

## Generate from an isolated cache

Transformers cache objects grow as generation proceeds. Passing the preserved object directly would make the result depend on which request ran first. The utility deep-copies the cache for each call:

```python
request_cache = copy.deepcopy(prefix_cache)

with torch.inference_mode():
    outputs = model.generate(
        **full_inputs,
        past_key_values=request_cache,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
```

Passing the full request follows the current Transformers prefix-caching example: `generate()` receives the complete attention mask and the prefilled cache, then prepares the uncached input positions internally. If you write a custom token-by-token generation loop instead, you must maintain both the combined attention-mask length and `cache_position` yourself.

Deep copying is deliberately part of the example, not a claim that it is the cheapest possible design. It is a straightforward way to demonstrate request isolation. For a particular model and cache class, an explicit clone or reset operation may be more appropriate.

## Run the utility

The repository script targets `transformers==4.57.3` and the public `Qwen/Qwen2-0.5B-Instruct` checkpoint. Qwen’s model card requires Transformers 4.37 or newer; pinning one later version makes the example easier to reproduce.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "torch>=2.2" "transformers==4.57.3"

python scripts/transformers_context_caching.py \
  --prompt "What do I need to request a refund?" \
  --prompt "My order arrived damaged. What should I do?"
```

The default run performs the useful check first: greedy cached and uncached generation must produce identical token IDs for each prompt. It prints the model, device, PyTorch version, Transformers version, and cached prefix length so a result is not separated from its environment.

To collect a small local timing sample:

```bash
python scripts/transformers_context_caching.py \
  --prompt "What do I need to request a refund?" \
  --benchmark-repeats 5
```

The timed cached path includes `deepcopy(prefix_cache)`. On CUDA, the utility synchronizes the device before and after each timed call because GPU execution is asynchronous. It reports medians, but the output is still a local measurement—not a portable benchmark.

## What to test before using this pattern

Greedy token parity is a strong first check, but it is not a complete production test suite. For the exact model and Transformers version you plan to use, also test:

- several prompts with different token lengths;
- an empty or whitespace-only request if your application permits it;
- a request containing a unique marker, followed by another request that must not reproduce it;
- maximum context limits after adding the fixed prefix and requested output length;
- cache-copy time and cache memory at the prefix sizes you actually expect;
- batch sizes greater than one, if the application batches requests;
- the failure path for a tokenizer whose chat template does not preserve the selected boundary.

Sampled generation needs a different equivalence test because independently sampled calls can produce different tokens even when their probability distributions are valid. Start with greedy decoding to validate the cache plumbing.

## When this is useful

This utility is useful for a direct-Transformers experiment where:

- many independent requests share a substantial fixed prefix;
- you control the model, tokenizer, and library version;
- a single-process cache is enough; and
- you want to inspect correctness before introducing a serving system.

It is a poor substitute for a production scheduler. Inference engines such as vLLM expose automatic prefix caching and manage cache blocks across concurrent requests. A serving layer may also handle batching, eviction, memory pressure, and request scheduling—concerns this script intentionally leaves out.

## Limits of the example

- It targets one model and one pinned Transformers version.
- It requires a chat template that accepts a system message and preserves the verified prefix.
- A deep copy may cost enough time or memory to offset the saved prefix computation.
- Identical token prefixes do not guarantee a latency improvement; measure on the target hardware and workload.
- The repository’s offline tests cover the token-boundary guard, request-local cache copy, and output slicing with test doubles. Full generation requires downloading the model and running the integration command above.

## References

- [Hugging Face: prefill a cache (prefix caching)](https://huggingface.co/docs/transformers/v4.57.3/en/kv_cache#prefill-a-cache-prefix-caching)
- [Hugging Face: cache position and attention-mask requirements](https://huggingface.co/docs/transformers/v4.57.3/en/cache_explanation)
- [Hugging Face: chat templates](https://huggingface.co/docs/transformers/v4.57.3/en/chat_templating)
- [Qwen2-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- [PyTorch: inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
- [PyTorch: CUDA timing and synchronization](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
- [vLLM: automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

The complete utility is available in [the repository](https://github.com/ToluClassics/toluclassics.github.io/blob/master/scripts/transformers_context_caching.py).
