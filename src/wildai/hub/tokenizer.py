"""Convert the training tokenizer to a Hugging Face ``tokenizer.json`` and check that the two agree.

The training tokenizer is a tiktoken ``Encoding`` (byte-level BPE, 32,768 tokens including 9 special tokens) pickled as
``tokenizer.pkl``. The converted tokenizer has the same vocabulary, merge order and pre-tokenization regex, so it
returns the same token ids. Like the training data loader, it prepends ``<|bos|>`` to every text by default
(``add_special_tokens=False`` turns that off).

Run: python -m wildai.hub.tokenizer --tiktoken path/to/tokenizer.pkl --out DIR [--check texts.parquet ...]
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from wildai.training.tokenizer import byte_to_unicode

BOS = "<|bos|>"


def load_encoding(path: Path) -> tiktoken.Encoding:
    """Unpickle the training tokenizer (only load files you trust: unpickling runs code)."""
    with path.open("rb") as f:
        encoding = pickle.load(f)
    if not isinstance(encoding, tiktoken.Encoding):
        raise TypeError(f"{path} does not hold a tiktoken Encoding")
    return encoding


def to_hf_tokenizer(encoding: tiktoken.Encoding) -> Tokenizer:
    """Byte-level BPE equivalent to ``encoding``.

    tiktoken merges, at every step, the adjacent pair whose concatenation has the lowest rank. Listing every split of every
    token as a merge, ordered by the rank of the merged token, makes the Hugging Face BPE pick the same pair. tiktoken also
    returns a whole pre-token directly when it is in the vocabulary, which ``ignore_merges`` reproduces.
    """
    byte_map = byte_to_unicode()
    specials = sorted(encoding.special_tokens_set, key=encoding.encode_single_token)
    n_regular = encoding.n_vocab - len(specials)
    ranks = {encoding.decode_single_token_bytes(i): i for i in range(n_regular)}

    def as_text(token: bytes) -> str:
        return "".join(byte_map[b] for b in token)

    merges: list[tuple[int, int, int, str, str]] = []
    for token, rank in ranks.items():
        for cut in range(1, len(token)):
            left, right = token[:cut], token[cut:]
            if left in ranks and right in ranks:
                merges.append((rank, ranks[left], ranks[right], as_text(left), as_text(right)))
    merges.sort()
    model = BPE(
        vocab={as_text(token): rank for token, rank in ranks.items()},
        merges=[(left, right) for *_, left, right in merges],
        ignore_merges=True,
        fuse_unk=False,
        byte_fallback=False,
    )
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(encoding._pat_str), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in specials])
    for token in specials:
        if tokenizer.token_to_id(token) != encoding.encode_single_token(token):
            raise ValueError(f"special token {token} got a different id")
    bos_id = encoding.encode_single_token(BOS)
    tokenizer.post_processor = processors.Sequence(
        [
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(single=f"{BOS} $A", pair=f"{BOS} $A {BOS} $B:1", special_tokens=[(BOS, bos_id)]),
        ]
    )
    return tokenizer


def save_hf_tokenizer(tokenizer: Tokenizer, out_dir: Path, model_max_length: int) -> None:
    """Write tokenizer.json and tokenizer_config.json for ``AutoTokenizer``.

    <|bos|> is also the end-of-text token: training documents were delimited by <|bos|> alone. The generic
    ``PreTrainedTokenizerFast`` class keeps the files loadable across transformers versions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": BOS,
        "eos_token": BOS,
        "model_max_length": model_max_length,
        "model_input_names": ["input_ids", "attention_mask"],
        "clean_up_tokenization_spaces": False,
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")


@dataclass(frozen=True)
class Agreement:
    """Token-id agreement between the training tokenizer and the converted one over a set of texts."""

    texts: int
    tokens: int
    mismatched_texts: int
    failed_round_trips: int
    first_mismatch: str | None

    @property
    def ok(self) -> bool:
        return self.mismatched_texts == 0 and self.failed_round_trips == 0


def check_agreement(encoding: tiktoken.Encoding, tokenizer: Tokenizer, texts: Iterable[str]) -> Agreement:
    """Compare both ways special-token strings inside text can be handled, and the decode round trip.

    Training encoded documents with ``encode_ordinary`` (special-token strings are plain text) and prepended <|bos|>;
    that matches the converted tokenizer with ``split_special_tokens=True``. Its default instead parses special-token
    strings, like tiktoken's ``encode(allowed_special="all")``.
    """
    bos_id = encoding.encode_single_token(BOS)
    plain = Tokenizer.from_str(tokenizer.to_str())
    plain.encode_special_tokens = True
    n_texts = n_tokens = mismatched = failed = 0
    first: str | None = None
    for text in texts:
        n_texts += 1
        ordinary = [bos_id, *encoding.encode_ordinary(text)]
        parsed = [bos_id, *encoding.encode(text, allowed_special="all")]
        n_tokens += len(ordinary)
        if plain.encode(text).ids != ordinary or tokenizer.encode(text).ids != parsed:
            mismatched += 1
            first = first if first is not None else text[:200]
        if tokenizer.decode(ordinary[1:], skip_special_tokens=False) != text:
            failed += 1
    return Agreement(texts=n_texts, tokens=n_tokens, mismatched_texts=mismatched, failed_round_trips=failed, first_mismatch=first)


def parquet_texts(paths: Iterable[Path], per_file: int) -> Iterable[str]:
    """Up to ``per_file`` texts from the ``text`` column of each parquet file."""
    for path in paths:
        table = pq.read_table(path, columns=["text"])
        yield from table.column("text").to_pylist()[:per_file]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiktoken", type=Path, required=True, help="the training tokenizer.pkl")
    parser.add_argument("--out", type=Path, required=True, help="directory for tokenizer.json and tokenizer_config.json")
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--check", type=Path, nargs="*", default=[], help="parquet files with a text column to compare ids on")
    parser.add_argument("--check-per-file", type=int, default=1000)
    args = parser.parse_args()
    encoding = load_encoding(args.tiktoken)
    tokenizer = to_hf_tokenizer(encoding)
    save_hf_tokenizer(tokenizer, args.out, args.model_max_length)
    print(f"wrote {args.out}")
    if args.check:
        agreement = check_agreement(encoding, tokenizer, parquet_texts(args.check, args.check_per_file))
        print(agreement)
        if not agreement.ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
