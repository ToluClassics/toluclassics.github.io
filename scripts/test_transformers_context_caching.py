import importlib.util
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).with_name("transformers_context_caching.py")
SPEC = importlib.util.spec_from_file_location(
    "transformers_context_caching",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
context_caching = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(context_caching)

PrefixCacheRunner = context_caching.PrefixCacheRunner
require_exact_prefix = context_caching.require_exact_prefix


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, columns = key
            return FakeTensor([self.rows[row][columns]])
        return FakeRow(self.rows[key])

    def tolist(self):
        if len(self.rows) == 1:
            return list(self.rows[0])
        return [list(row) for row in self.rows]


class FakeRow:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    prefix_ids = [10, 11]
    request_ids = [10, 11, 20, 21]

    def apply_chat_template(self, messages, **_kwargs):
        token_ids = self.prefix_ids if len(messages) == 1 else self.request_ids
        return FakeBatch(
            input_ids=FakeTensor([token_ids]),
            attention_mask=FakeTensor([[1] * len(token_ids)]),
        )

    def decode(self, token_ids, **_kwargs):
        return ",".join(str(token_id) for token_id in token_ids)


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeModel:
    def __init__(self):
        self.device = SimpleNamespace(type="cpu")
        self.generate_calls = []

    def __call__(self, **_kwargs):
        return SimpleNamespace(past_key_values={"cached": [10, 11]})

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        prompt_ids = kwargs["input_ids"].rows[0]
        return FakeTensor([prompt_ids + [99]])


class ExactTokenPrefixTest(unittest.TestCase):
    def test_accepts_exact_prefix(self):
        self.assertEqual(require_exact_prefix([11, 12], [11, 12, 13]), 2)

    def test_rejects_token_mismatch(self):
        with self.assertRaisesRegex(ValueError, "not an exact token prefix"):
            require_exact_prefix([11, 99], [11, 12, 13])

    def test_rejects_empty_prefix(self):
        with self.assertRaisesRegex(ValueError, "contains no tokens"):
            require_exact_prefix([], [11, 12, 13])

    def test_rejects_prefix_longer_than_request(self):
        with self.assertRaisesRegex(ValueError, "not an exact token prefix"):
            require_exact_prefix([11, 12, 13], [11, 12])


class PrefixCacheRunnerTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.runner = PrefixCacheRunner(
            model=self.model,
            tokenizer=FakeTokenizer(),
            torch_module=FakeTorch(),
            context="fixed context",
            max_new_tokens=1,
        )

    def test_cached_call_uses_full_verified_request_and_cache_copy(self):
        generated_ids = self.runner.generate_cached("new request")

        call = self.model.generate_calls[-1]
        self.assertEqual(call["input_ids"].rows, [[10, 11, 20, 21]])
        self.assertEqual(call["attention_mask"].rows, [[1, 1, 1, 1]])
        self.assertEqual(call["past_key_values"], {"cached": [10, 11]})
        self.assertIsNot(call["past_key_values"], self.runner.prefix_cache)
        self.assertEqual(generated_ids, [99])

    def test_cached_and_uncached_slicing_returns_same_generated_ids(self):
        self.assertEqual(
            self.runner.generate_cached("new request"),
            self.runner.generate_uncached("new request"),
        )


if __name__ == "__main__":
    unittest.main()
