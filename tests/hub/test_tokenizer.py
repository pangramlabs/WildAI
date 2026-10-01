from __future__ import annotations

import os
from pathlib import Path

import pytest
import tiktoken
from transformers import AutoTokenizer

from wildai.hub.tokenizer import BOS, check_agreement, load_encoding, save_hf_tokenizer, to_hf_tokenizer


def test_converted_tokenizer_matches_tiktoken(encoding: tiktoken.Encoding, sample_texts: list[str]) -> None:
    agreement = check_agreement(encoding, to_hf_tokenizer(encoding), sample_texts)
    assert agreement.ok, agreement
    assert agreement.tokens > 1000


def test_special_tokens_keep_their_ids(encoding: tiktoken.Encoding) -> None:
    tokenizer = to_hf_tokenizer(encoding)
    for token in encoding.special_tokens_set:
        assert tokenizer.token_to_id(token) == encoding.encode_single_token(token)


def test_auto_tokenizer_prepends_bos_and_round_trips(encoding: tiktoken.Encoding, tmp_path: Path) -> None:
    save_hf_tokenizer(to_hf_tokenizer(encoding), tmp_path, model_max_length=512)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    bos = encoding.encode_single_token(BOS)
    text = "Numbers 12345 and naïve café.\n\n  def f(): pass"
    assert tokenizer.bos_token_id == tokenizer.eos_token_id == bos
    assert tokenizer(text).input_ids == [bos, *encoding.encode_ordinary(text)]
    assert tokenizer(text, add_special_tokens=False).input_ids == encoding.encode_ordinary(text)
    assert tokenizer("a <|bos|> b", split_special_tokens=True).input_ids == [bos, *encoding.encode_ordinary("a <|bos|> b")]
    assert tokenizer.decode(tokenizer(text).input_ids, skip_special_tokens=True) == text


@pytest.mark.slow
@pytest.mark.skipif("WILDAI_TOKENIZER" not in os.environ, reason="set WILDAI_TOKENIZER to the training tokenizer.pkl")
def test_training_tokenizer_agreement(sample_texts: list[str]) -> None:
    encoding = load_encoding(Path(os.environ["WILDAI_TOKENIZER"]))
    assert encoding.n_vocab == 32768
    assert encoding.encode_single_token(BOS) == 32759
    agreement = check_agreement(encoding, to_hf_tokenizer(encoding), sample_texts)
    assert agreement.ok, agreement
