"""Tests for the T5 code encoder, tokenization, and parameter masking."""

from pathlib import Path
import string

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from peace import PEACE, PEACEConfig, T5Config
from peace.audio import AudioEncoderConfig
from peace.graph import parameter_spans
from peace.model import flat_weights
from peace.t5 import relative_position_buckets


def character_tokenizer() -> Tokenizer:
    """A SentencePiece-style unigram tokenizer with one token per character."""
    pieces = ["<pad>", "</s>", "<unk>", "▁", *string.printable.strip()]
    tokenizer = Tokenizer(models.Unigram([(p, -1.0) for p in pieces], unk_id=2))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", 1)]
    )
    return tokenizer


@pytest.fixture(scope="module")
def model() -> PEACE:
    code = T5Config(vocab_size=128, d_model=16, head_dim=4, d_ff=32, num_layers=2, num_heads=2)
    return PEACE(
        PEACEConfig(
            audio=AudioEncoderConfig(channels=(4, 4, 4, 4, 4, 8), embed_dim=8),
            code=code,
            embed_dim=8,
            predictor_hidden_dim=16,
        ),
        tokenizer=character_tokenizer(),
    )


def program(model: PEACE, chain) -> str:
    return model.library.program(chain)


def test_program_matches_training_format(model: PEACE) -> None:
    """Programs are written as the training data generator wrote them."""
    code = model.library.program([("filter", [10, 0.8583, 0.3436, 0.6717, 0.6758, 0.1204])])
    assert code == (
        "box1 = filter(\n    10,  // mode\n    0.8583,  // cutoff\n    0.3436,  // res\n"
        "    0.6717,  // drive\n    0.6758,  // other\n    0.1204  // mix\n);\n\n"
        "process = box1;\n"
    )
    chain = model.library.chain(code)
    assert model.library.program(chain) == code


def test_relative_position_buckets_match_t5() -> None:
    """Near offsets get their own buckets, far ones share log-spaced buckets."""
    buckets = relative_position_buckets(300, 32, 128)
    assert buckets[0, 0] == 0
    assert buckets[0, 7] == 16 + 7 and buckets[7, 0] == 7
    assert buckets[0, 8] == 16 + 8 and buckets[8, 0] == 8
    assert buckets[0, 299] == 31 and buckets[299, 0] == 15
    assert buckets.max() == 31


def test_tokenizer_rounds_and_masks_parameters(model: PEACE) -> None:
    """Decimals are rounded before tokenizing; masking hides only their tokens."""
    tokenizer = model.code_encoder.tokenizer
    code = "process = stereo_width(0.25678);  // width 0.5"
    rounded = tokenizer.round(code)
    assert rounded == "process = stereo_width(0.26);  // width 0.50"
    assert parameter_spans(rounded) == [(23, 27)]
    visible = tokenizer([code], mask_parameters=False)
    masked = tokenizer([code], mask_parameters=True)
    np.testing.assert_array_equal(visible.input_ids, masked.input_ids)
    count = visible.attention_mask.sum()
    assert masked.attention_mask.sum() == count - 4
    assert visible.input_ids.shape[1] % 64 == 0


def test_tokens_are_truncated(model: PEACE) -> None:
    code = program(model, [("stereo_width", [0.5])] * 60)
    tokens = model.code_encoder.tokenizer([code], mask_parameters=False)
    assert tokens.input_ids.shape[1] == model.config.code.max_length
    assert tokens.attention_mask.sum() == model.config.code.max_length


def test_padding_does_not_change_embeddings(model: PEACE) -> None:
    """Short programs batched with long ones embed as they do alone."""
    short = program(model, [("stereo_width", [0.25])])
    long = program(model, [("delay", [1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])] * 3)
    alone = model.encode_code(short)
    together = model.encode_code([long, short])
    np.testing.assert_allclose(together[1:], alone, atol=1e-5)


def test_mask_parameters_hides_values(model: PEACE) -> None:
    """Masked embeddings ignore values of the same length but not effects."""
    first = program(model, [("stereo_width", [0.25])])
    other = program(model, [("stereo_width", [0.75])])
    different = program(model, [("stereo_width", [0.25]), ("stereo_width", [0.25])])
    masked = model.embed_code([first, other, different], mask_parameters=True).latent
    np.testing.assert_allclose(masked[0], masked[1], atol=1e-5)
    assert np.abs(masked[0] - masked[2]).max() > 1e-3
    visible = model.embed_code([first, other]).latent
    assert np.abs(visible[0] - visible[1]).max() > 1e-4


def test_programs_are_checked(model: PEACE) -> None:
    with pytest.raises(ValueError, match="Unknown effect"):
        model.encode_code("process = missing(0.5);")


def test_constructor_requires_tokenizer() -> None:
    with pytest.raises(ValueError, match="tokenizer"):
        PEACE(PEACEConfig(code=T5Config(num_layers=1)))


def test_round_trip_keeps_tokenizer(model: PEACE, tmp_path: Path) -> None:
    model.save_pretrained(tmp_path)
    assert (tmp_path / "tokenizer.json").exists()
    restored = PEACE.from_pretrained(tmp_path)
    assert isinstance(restored.config.code, T5Config)
    original, actual = flat_weights(model), flat_weights(restored)
    assert original.keys() == actual.keys()
    code = program(model, [("stereo_width", [0.25])])
    for masked in (False, True):
        np.testing.assert_allclose(
            restored.encode_code(code, mask_parameters=masked),
            model.encode_code(code, mask_parameters=masked),
            atol=1e-6,
        )
