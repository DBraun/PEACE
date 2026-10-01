"""T5 code encoder: a T5 v1.1 encoder over Faust source text, mean-pooled."""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence
import re

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from tokenizers import Tokenizer

from .graph import EffectLibrary, parameter_spans

MASKED_LOGIT = -1e10


@dataclass
class T5Config:
    """T5 encoder and tokenization configuration; defaults are T5 v1.1 small.

    Attributes:
        vocab_size: SentencePiece vocabulary size.
        d_model: Token feature width and output width.
        head_dim: Width of each attention head.
        d_ff: Hidden width of the gated-GELU feed-forward layers.
        num_layers: Encoder blocks.
        num_heads: Attention heads.
        num_buckets: Relative position buckets, half for each direction.
        max_distance: Distance beyond which relative positions share a bucket.
        epsilon: RMS normalization epsilon.
        max_length: Programs are truncated to this many tokens.
        decimal_places: Decimal literals are rounded to this many places before
            tokenization.
    """

    vocab_size: int = 32128
    d_model: int = 512
    head_dim: int = 64
    d_ff: int = 1024
    num_layers: int = 8
    num_heads: int = 6
    num_buckets: int = 32
    max_distance: int = 128
    epsilon: float = 1e-6
    max_length: int = 588
    decimal_places: int = 2


class Tokens(NamedTuple):
    """Token IDs and attention mask, both ``[batch, length]``."""

    input_ids: np.ndarray
    attention_mask: np.ndarray


class CodeTokenizer:
    """Round decimals, tokenize, and optionally hide parameter-value tokens."""

    def __init__(self, tokenizer: Tokenizer, config: T5Config) -> None:
        self.tokenizer = tokenizer
        self.config = config
        tokenizer.enable_truncation(config.max_length)
        tokenizer.no_padding()

    @classmethod
    def from_file(cls, path: str | Path, config: T5Config) -> "CodeTokenizer":
        return cls(Tokenizer.from_file(str(path)), config)

    def save(self, path: str | Path) -> None:
        self.tokenizer.save(str(path))

    def round(self, code: str) -> str:
        places = self.config.decimal_places
        return re.sub(r"\d+\.\d+", lambda m: f"{float(m.group(0)):.{places}f}", code)

    def __call__(self, programs: Sequence[str], mask_parameters: bool) -> Tokens:
        """Tokenize programs, padded to a multiple of 64 tokens.

        With ``mask_parameters``, tokens that overlap a numeric argument are
        excluded from attention and pooling, so the embedding sees effect names,
        comments, and structure but no parameter values.
        """
        texts = [self.round(program) for program in programs]
        encodings = self.tokenizer.encode_batch(texts)
        longest = max(len(e.ids) for e in encodings)
        length = min(self.config.max_length, -(-longest // 64) * 64)
        input_ids = np.zeros((len(texts), length), np.int32)
        attention_mask = np.zeros((len(texts), length), np.int32)
        for i, (text, encoding) in enumerate(zip(texts, encodings)):
            count = len(encoding.ids)
            input_ids[i, :count] = encoding.ids
            keep = np.ones(count, np.int32)
            if mask_parameters:
                spans = parameter_spans(text)
                for j, (start, end) in enumerate(encoding.offsets):
                    if start < end and any(max(start, a) < min(end, b) for a, b in spans):
                        keep[j] = 0
            attention_mask[i, :count] = keep
        return Tokens(input_ids, attention_mask)


def relative_position_buckets(
    length: int, num_buckets: int, max_distance: int
) -> np.ndarray:
    """Bidirectional T5 relative position buckets, ``[length, length]``.

    Half the buckets hold keys after the query. Within each half, distances
    below a quarter of ``num_buckets`` get their own bucket and larger
    distances share logarithmically spaced buckets up to ``max_distance``.
    """
    position = np.arange(length, dtype=np.int32)
    relative = position[None, :] - position[:, None]
    half = num_buckets // 2
    exact = half // 2
    distance = np.abs(relative)
    large = exact + (
        np.log(distance.astype(np.float32) / exact + np.finfo(np.float32).eps)
        / np.log(max_distance / exact)
        * (half - exact)
    ).astype(np.int32)
    large = np.minimum(large, half - 1)
    return (relative > 0) * half + np.where(distance < exact, distance, large)


class T5Block(nnx.Module):
    """Pre-norm self-attention and gated-GELU feed-forward, both residual.

    As in T5, attention logits are not scaled by the head width.
    """

    def __init__(self, config: T5Config, rngs: nnx.Rngs) -> None:
        width = config.num_heads * config.head_dim
        self.num_heads = config.num_heads
        self.attention_norm = nnx.RMSNorm(config.d_model, epsilon=config.epsilon, rngs=rngs)
        self.q = nnx.Linear(config.d_model, width, use_bias=False, rngs=rngs)
        self.k = nnx.Linear(config.d_model, width, use_bias=False, rngs=rngs)
        self.v = nnx.Linear(config.d_model, width, use_bias=False, rngs=rngs)
        self.o = nnx.Linear(width, config.d_model, use_bias=False, rngs=rngs)
        self.mlp_norm = nnx.RMSNorm(config.d_model, epsilon=config.epsilon, rngs=rngs)
        self.wi_0 = nnx.Linear(config.d_model, config.d_ff, use_bias=False, rngs=rngs)
        self.wi_1 = nnx.Linear(config.d_model, config.d_ff, use_bias=False, rngs=rngs)
        self.wo = nnx.Linear(config.d_ff, config.d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, bias: jax.Array) -> jax.Array:
        batch, length, _ = x.shape
        h = self.attention_norm(x)
        q, k, v = (
            layer(h).reshape(batch, length, self.num_heads, -1)
            for layer in (self.q, self.k, self.v)
        )
        weights = jax.nn.softmax(jnp.einsum("bqhd,bkhd->bhqk", q, k) + bias)
        attended = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(batch, length, -1)
        x = x + self.o(attended)
        h = self.mlp_norm(x)
        return x + self.wo(nnx.gelu(self.wi_0(h), approximate=True) * self.wi_1(h))


class T5CodeEncoder(nnx.Module):
    """Embed Faust source by mean-pooling T5 encoder states over attended tokens."""

    def __init__(
        self,
        config: T5Config,
        library: EffectLibrary,
        tokenizer: CodeTokenizer,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.library = library
        self.tokenizer = tokenizer
        self.embed = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)
        self.relative_bias = nnx.Param(
            nnx.initializers.normal(1.0)(rngs.params(), (config.num_heads, config.num_buckets))
        )
        self.blocks = nnx.List([T5Block(config, rngs) for _ in range(config.num_layers)])
        self.final_norm = nnx.RMSNorm(config.d_model, epsilon=config.epsilon, rngs=rngs)

    @property
    def out_features(self) -> int:
        return self.config.d_model

    def prepare(self, programs: Sequence[str], mask_parameters: bool) -> tuple[Tokens, tuple]:
        """Check programs against the effect library, then tokenize them."""
        for program in programs:
            self.library.parse(program)
        return self.tokenizer(programs, mask_parameters), ()

    def __call__(self, tokens: Tokens) -> jax.Array:
        """Return ``[batch, d_model]`` program embeddings."""
        cfg = self.config
        length = tokens.input_ids.shape[1]
        buckets = relative_position_buckets(length, cfg.num_buckets, cfg.max_distance)
        mask = tokens.attention_mask
        bias = self.relative_bias[...][:, buckets][None] + jnp.where(
            mask[:, None, None, :] > 0, 0.0, MASKED_LOGIT
        )
        x = self.embed(tokens.input_ids)
        for block in self.blocks:
            x = block(x, bias)
        x = self.final_norm(x)
        weights = mask[:, :, None].astype(x.dtype)
        return (x * weights).sum(axis=1) / weights.sum(axis=1)
