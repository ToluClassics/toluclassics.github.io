#!/usr/bin/env python3
"""Validate and optionally benchmark a reusable Transformers prefix cache.

The example targets Transformers 4.57.3 and Qwen/Qwen2-0.5B-Instruct. It keeps
the fixed context in a system message, verifies that its token IDs are an exact
prefix of every full request, and deep-copies the prefilled cache per request.
"""

from __future__ import annotations

import argparse
import copy
from statistics import median
import time
from typing import Any, Sequence


DEFAULT_CONTEXT = """You answer questions using this support policy:
- A refund is available within 30 days of purchase.
- The customer must provide an order number.
- Escalate damaged-item claims to a human agent.
If the policy does not answer the question, say that you need more information."""

DEFAULT_PROMPTS = [
    "What do I need to request a refund?",
    "My order arrived damaged. What should I do?",
]


def require_exact_prefix(prefix_ids: Sequence[int], full_ids: Sequence[int]) -> int:
    """Return the prefix length or fail before an invalid cache is used."""
    prefix_length = len(prefix_ids)
    if prefix_length == 0:
        raise ValueError("The rendered reusable prefix contains no tokens.")
    if prefix_length > len(full_ids) or list(full_ids[:prefix_length]) != list(prefix_ids):
        raise ValueError(
            "The rendered reusable context is not an exact token prefix of the "
            "full request. This chat template cannot use the selected boundary."
        )
    return prefix_length


class PrefixCacheRunner:
    """Generate independent responses from one immutable prefilled cache."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        context: str,
        max_new_tokens: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.context = context
        self.max_new_tokens = max_new_tokens
        self.prefix_cache, self.prefix_ids = self._build_prefix_cache()

    def _messages(self, prompt: str | None = None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.context}]
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
        return messages

    def _tokenize_chat(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> Any:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        return encoded.to(self.model.device)

    def _build_prefix_cache(self) -> tuple[Any, list[int]]:
        prefix_inputs = self._tokenize_chat(
            self._messages(),
            add_generation_prompt=False,
        )
        prefix_ids = prefix_inputs["input_ids"][0].tolist()
        with self.torch.inference_mode():
            output = self.model(**prefix_inputs, use_cache=True)
        return output.past_key_values, prefix_ids

    def _full_inputs(self, prompt: str) -> Any:
        inputs = self._tokenize_chat(
            self._messages(prompt),
            add_generation_prompt=True,
        )
        require_exact_prefix(self.prefix_ids, inputs["input_ids"][0].tolist())
        return inputs

    def _generated_ids(self, outputs: Any, prompt_length: int) -> list[int]:
        return outputs[0, prompt_length:].tolist()

    def generate_uncached(self, prompt: str) -> list[int]:
        inputs = self._full_inputs(prompt)
        prompt_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        return self._generated_ids(outputs, prompt_length)

    def generate_cached(self, prompt: str) -> list[int]:
        inputs = self._full_inputs(prompt)
        prompt_length = inputs["input_ids"].shape[1]
        try:
            request_cache = copy.deepcopy(self.prefix_cache)
        except Exception as error:
            raise RuntimeError(
                "This model's cache cannot be deep-copied. Use a cache "
                "implementation with an explicit clone/reset operation."
            ) from error

        with self.torch.inference_mode():
            # Transformers 4.57.3 uses the cache length to trim already
            # processed positions from these full input IDs inside
            # prepare_inputs_for_generation(). Keeping the full input here also
            # keeps the returned sequence and prompt-length slicing consistent
            # with the uncached path.
            outputs = self.model.generate(
                **inputs,
                past_key_values=request_cache,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        return self._generated_ids(outputs, prompt_length)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def synchronize(self) -> None:
        if self.model.device.type == "cuda":
            self.torch.cuda.synchronize(self.model.device)

    def timed(self, fn: Any, prompt: str) -> tuple[list[int], float]:
        self.synchronize()
        started = time.perf_counter()
        result = fn(prompt)
        self.synchronize()
        return result, time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check correctness and measure a reusable Transformers prefix cache."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Causal chat model with system-message support.",
    )
    parser.add_argument(
        "--context",
        default=DEFAULT_CONTEXT,
        help="Fixed context reused across all requests.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Independent user request. Repeat this option for multiple requests.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=0,
        help="Timed repeats per path after one warm-up; 0 runs correctness only.",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.benchmark_repeats < 0:
        parser.error("--benchmark-repeats cannot be negative")
    if not args.prompts:
        args.prompts = DEFAULT_PROMPTS
    return args


def load_runtime(model_id: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            'Install the example dependencies first: pip install "torch>=2.2" '
            '"transformers==4.57.3"'
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, torch


def main() -> None:
    args = parse_args()
    model, tokenizer, torch = load_runtime(args.model_id)
    runner = PrefixCacheRunner(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        context=args.context,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"model={args.model_id}")
    print(f"device={model.device}")
    print(f"torch={torch.__version__}")
    import transformers

    print(f"transformers={transformers.__version__}")
    print(f"cached_prefix_tokens={len(runner.prefix_ids)}")

    for prompt in args.prompts:
        uncached_ids = runner.generate_uncached(prompt)
        cached_ids = runner.generate_cached(prompt)
        if cached_ids != uncached_ids:
            raise RuntimeError(
                "Greedy cached and uncached generations differ for prompt: "
                f"{prompt!r}"
            )
        print(f"\nprompt: {prompt}")
        print(f"response: {runner.decode(cached_ids)}")
        print("token_parity=pass")

    if args.benchmark_repeats == 0:
        return

    benchmark_prompt = args.prompts[0]
    runner.generate_uncached(benchmark_prompt)
    runner.generate_cached(benchmark_prompt)
    uncached_times = []
    cached_times = []
    for _ in range(args.benchmark_repeats):
        _, uncached_time = runner.timed(runner.generate_uncached, benchmark_prompt)
        _, cached_time = runner.timed(runner.generate_cached, benchmark_prompt)
        uncached_times.append(uncached_time)
        cached_times.append(cached_time)

    print(f"\nbenchmark_prompt: {benchmark_prompt}")
    print(f"repeats={args.benchmark_repeats}")
    print(f"uncached_median_seconds={median(uncached_times):.6f}")
    print(f"cached_median_seconds={median(cached_times):.6f}")
    print("Cached timing includes deepcopy of the reusable cache.")


if __name__ == "__main__":
    main()
