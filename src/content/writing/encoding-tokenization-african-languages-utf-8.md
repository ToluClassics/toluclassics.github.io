---
title: "Encoding, Tokenization, and African Languages: Why UTF-8 Matters"
description: "How Unicode encoding, normalization, and byte-level tokenization interact across Swahili, Yorùbá, and Amharic text."
publishedAt: 2025-09-29
updatedAt: 2026-07-25
draft: false
topics: ["African NLP", "Tokenization"]
featured: true
readingMinutes: 12
---

Working on multilingual NLP for African languages has made me increasingly curious about text encoding. I became especially interested in how byte-oriented tokenizers handle the range of writing systems used across the continent.

That question started after I read [UTF-8 is a Brilliant Design](https://iamvishnu.com/posts/utf8-is-brilliant-design/). The article explains the mechanics of UTF-8 beautifully. I wanted to take the next step: what does that design mean for African-language text, and where does encoding stop and tokenization begin?

This article uses three short examples—Swahili, Yorùbá, and Amharic—to work through that boundary.

> This is a revised edition of an article first published by [The African Research Collective](https://medium.com/the-african-research-collective/encoding-tokenization-and-african-languages-why-utf-8-matters-6dd3318d0240) on September 29, 2025.

## ASCII stops at 128 code points

ASCII uses seven bits to represent 128 values. That covers the basic Latin letters used in English, digits, punctuation, and control characters. It cannot represent most of Unicode.

The limitation is immediate with a Yorùbá greeting:

```python
text = "Ẹ káàárọ̀"
text.encode("ascii")
```

Python raises `UnicodeEncodeError` because characters such as `Ẹ`, `á`, and `ọ` are outside ASCII.

The issue is not that the text is unusual. Yorùbá uses diacritics to represent distinctions that matter to the language. Dropping those marks to fit an older encoding changes the text.

## Three examples from different writing systems

African languages do not fit into one script or one encoding profile. These three strings are useful examples, not an exhaustive classification:

1. **Swahili:** `Habari za asubuhi` uses only ASCII code points in this particular sentence.
2. **Yorùbá:** `Ẹ káàárọ̀` uses Latin letters, precomposed diacritics, and a combining grave accent.
3. **Amharic:** `እንደምን አደሩ` uses the Ethiopic script.

Unicode assigns code points to all of them. UTF-8, UTF-16, and UTF-32 are different ways of encoding those code points as code units:

| Encoding | Code-unit width | Code units per Unicode scalar value |
| --- | --- | --- |
| UTF-8 | 8 bits | 1–4 |
| UTF-16 | 16 bits | 1 or 2 |
| UTF-32 | 32 bits | 1 |

UTF-8 and UTF-16 are both variable-width encodings. UTF-32 is fixed-width at the code-unit level.

UTF-8 preserves ASCII byte-for-byte: code points from U+0000 through U+007F use one byte with the same value as ASCII. Other code points use two, three, or four bytes according to their numeric range. This is about the code point value, not how visually or linguistically “complex” a character is.

## Compare the byte lengths

For a clean comparison, I use the explicit little-endian forms of UTF-16 and UTF-32 below. Python's `utf-16` and `utf-32` encoders include a byte-order mark, while `utf-16-le` and `utf-32-le` do not.

```python
examples = {
    "Swahili": "Habari za asubuhi",
    "Yorùbá": "Ẹ káàárọ̀",
    "Amharic": "እንደምን አደሩ",
}

for language, text in examples.items():
    print(
        language,
        len(text),
        len(text.encode("utf-8")),
        len(text.encode("utf-16-le")),
        len(text.encode("utf-32-le")),
    )
```

The result is:

| Example | Python code points | UTF-8 bytes | UTF-16LE bytes | UTF-32LE bytes |
| --- | ---: | ---: | ---: | ---: |
| `Habari za asubuhi` | 17 | 17 | 34 | 68 |
| `Ẹ káàárọ̀` | 9 | 17 | 18 | 36 |
| `እንደምን አደሩ` | 9 | 25 | 18 | 36 |

UTF-8 is the smallest representation for the ASCII-only Swahili example. UTF-16LE is smaller for the Amharic example because the Ethiopic code points in this string fit in one 16-bit code unit each. The Yorùbá example sits between them.

There is no contradiction here. Every Unicode encoding form can represent every string in the table. Their byte lengths differ because they use different code-unit widths and mapping rules.

## Yorùbá exposes a second problem: normalization

What looks like one written character is not always one Unicode code point. The final `ọ̀` in `Ẹ káàárọ̀` is represented here by:

```text
U+1ECD  LATIN SMALL LETTER O WITH DOT BELOW
U+0300  COMBINING GRAVE ACCENT
```

Unicode normalization can produce canonically equivalent strings with different code-point and byte sequences. In Python:

```python
import unicodedata

text = "Ẹ káàárọ̀"
nfc = unicodedata.normalize("NFC", text)
nfd = unicodedata.normalize("NFD", text)

print(len(nfc), len(nfc.encode("utf-8")))  # 9 code points, 17 bytes
print(len(nfd), len(nfd.encode("utf-8")))  # 14 code points, 20 bytes
print(nfc == nfd)                          # False
```

NFC and NFD are canonically equivalent, but Python string equality compares their code-point sequences directly. A tokenizer may also segment them differently unless its normalization stage makes them consistent first.

For African-language NLP, this is often more consequential than comparing UTF-8 with UTF-16. Two visually identical inputs can arrive with different underlying sequences because of keyboards, data sources, or preprocessing pipelines.

## Encoding is not tokenization

The original version of this article treated BPE as if it always began with raw bytes. That is true for **byte-level BPE**, but not for every BPE implementation.

A conventional BPE tokenizer can begin with characters or other symbols produced after normalization and pre-tokenization. A byte-level BPE tokenizer instead maps the UTF-8 bytes to a 256-symbol alphabet before learning merges. Hugging Face's `ByteLevel` pre-tokenizer follows this design.

Other tokenizer families make different choices:

- OpenAI's `tiktoken` uses byte sequences internally. A token boundary does not have to align with a UTF-8 character boundary.
- SentencePiece treats input as sequences of Unicode characters, applies normalization by default, and supports BPE and unigram models. It can optionally decompose unknown pieces into UTF-8 bytes with byte fallback.

The distinction matters because “the tokenizer uses UTF-8” does not fully describe its behavior. We also need to know:

- what normalization is applied;
- what the initial symbols are—bytes, characters, or pre-tokenized units;
- which merge or segmentation algorithm is used; and
- what languages and scripts were represented in the tokenizer's training corpus.

## Compression still matters

Byte-level tokenization guarantees coverage: any UTF-8 string can be represented without an unknown character. It does not guarantee that every language will be represented with the same number of tokens.

If a language or orthographic pattern appears frequently in the tokenizer's training data, the learned vocabulary has more opportunities to merge its common byte or character sequences. Underrepresented patterns may remain split into shorter pieces. The result is often measured as:

- **fertility:** the number of tokens per word; or
- **compression:** the number of bytes or characters represented per token.

These measures affect how much text fits in a context window and, for token-priced systems, how much equivalent text costs to process. They do not by themselves prove why a model performs better or worse in a language; training data, model capacity, morphology, and evaluation quality are confounding factors.

This is the point at which the African-language question becomes empirical. UTF-8 tells us that the text is representable. It does not tell us whether a particular tokenizer represents that text efficiently. That must be measured with the actual tokenizer and a representative corpus.

## Inspect a string before blaming the tokenizer

The repository includes a dependency-free inspection utility:

```bash
python scripts/inspect_text_encoding.py
python scripts/inspect_text_encoding.py "Ẹ káàárọ̀"
python scripts/inspect_text_encoding.py --json "እንደምን አደሩ"
```

It reports:

- code-point count and Unicode names;
- UTF-8, UTF-16LE, and UTF-32LE byte lengths; and
- NFC and NFD representations.

This does not replace a tokenizer benchmark. It answers the lower-level question first: what sequence did the tokenizer actually receive?

## What I take from this

UTF-8 is a strong default because it represents all of Unicode, preserves ASCII byte-for-byte, and works naturally with byte-oriented software. But UTF-8 alone does not make a tokenizer multilingual or fair.

For African-language systems, I would check three things separately:

1. **Encoding:** can the text move through the system without loss?
2. **Normalization:** do canonically equivalent inputs become consistent?
3. **Tokenization:** how efficiently does the target tokenizer represent real text from the language?

Conflating those layers makes tokenizer problems harder to diagnose. Separating them gives us measurements we can act on.

## References

- [The Unicode Standard: encoding forms](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-2/)
- [Unicode FAQ: UTF-8, UTF-16, UTF-32, and byte-order marks](https://www.unicode.org/faq/utf_bom.html)
- [Unicode Standard Annex #15: Normalization Forms](https://www.unicode.org/reports/tr15/)
- [Hugging Face: ByteLevel pre-tokenizer](https://huggingface.co/docs/tokenizers/main/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel)
- [Hugging Face course: BPE and byte-level BPE](https://huggingface.co/docs/course/chapter6/5)
- [`tiktoken` repository and byte-level decoding API](https://github.com/openai/tiktoken)
- [SentencePiece technical overview](https://github.com/google/sentencepiece)
- [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425)
- [The Token Tax: Systematic Bias in Multilingual Tokenization](https://arxiv.org/abs/2509.05486)
