"""Standalone inference for PEACE's audio and Faust code encoders."""

from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import NamedTuple, Sequence
import json

import jax
from jax import numpy as jnp
import numpy as np
from audiotree import AudioTree
from flax import nnx
from safetensors.numpy import load_file, save_file
from tokenizers import Tokenizer

from .audio import AudioEncoder, AudioEncoderConfig, l2_normalize
from .boxgraph import BoxGraphConfig, BoxGraphEncoder
from .graph import Effect, EffectLibrary, Parameter
from .t5 import CodeTokenizer, T5CodeEncoder, T5Config

WEIGHTS = nnx.Any(nnx.Param, nnx.BatchStat)
FORMAT_VERSION = 1
CODE_ENCODERS = {"boxgraph": BoxGraphConfig, "t5": T5Config}
TOKENIZER_FILE = "tokenizer.json"


def parse_effects(effects: list[dict]) -> tuple[Effect, ...]:
    """Build effects from their JSON form in ``effects.json`` or ``config.json``."""
    return tuple(
        Effect(e["name"], e["function"], tuple(Parameter(**p) for p in e["parameters"]))
        for e in effects
    )


def default_effects() -> tuple[Effect, ...]:
    """The 21-effect Faust library used to train the released model."""
    data = json.loads(resources.files(__package__).joinpath("effects.json").read_text())
    return parse_effects(data["effects"])


@dataclass
class PEACEConfig:
    """Architecture, preprocessing, and effect library for a PEACE model.

    Attributes:
        audio: Audio frontend and CNN14 configuration.
        code: BoxGraph or T5 code encoder configuration.
        embed_dim: Width of the shared projection and prediction spaces.
        predictor_hidden_dim: Hidden width of each predictor.
        effects: Effects whose names and parameters the code encoder knows.
    """

    audio: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    code: BoxGraphConfig | T5Config = field(default_factory=BoxGraphConfig)
    embed_dim: int = 768
    predictor_hidden_dim: int = 4096
    effects: tuple[Effect, ...] = field(default_factory=default_effects)


class Embeddings(NamedTuple):
    """Per-modality outputs of one encoder arm.

    Attributes:
        latent: Encoder output ``y``.
        projection: Unit-length projection ``z``.
        prediction: Unit-length prediction ``q``, used for retrieval.
    """

    latent: jax.Array
    projection: jax.Array
    prediction: jax.Array


class Heads(nnx.Module):
    """Projector and predictor MLPs that map an encoder latent to ``z`` and ``q``."""

    def __init__(self, in_dim: int, dim: int, hidden_dim: int, rngs: nnx.Rngs) -> None:
        self.projector_in = nnx.Linear(in_dim, dim, rngs=rngs)
        self.projector_out = nnx.Linear(dim, dim, rngs=rngs)
        self.predictor_in = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)
        self.predictor_norm = nnx.BatchNorm(hidden_dim, rngs=rngs)
        self.predictor_out = nnx.Linear(hidden_dim, dim, rngs=rngs)

    def __call__(self, latent: jax.Array) -> Embeddings:
        projection = self.projector_out(nnx.relu(self.projector_in(latent)))
        hidden = nnx.relu(self.predictor_norm(self.predictor_in(projection)))
        prediction = self.predictor_out(hidden)
        return Embeddings(latent, l2_normalize(projection), l2_normalize(prediction))


def flat_weights(model: nnx.Module) -> dict[str, np.ndarray]:
    """Return parameters and batch statistics keyed by their module paths."""
    pure = nnx.to_pure_dict(jax.device_get(nnx.state(model, WEIGHTS)))
    return {
        jax.tree_util.keystr(path): np.asarray(value)
        for path, value in jax.tree_util.tree_flatten_with_path(pure)[0]
    }


def load_weights(model: nnx.Module, weights: dict[str, np.ndarray]) -> None:
    """Load weights with exact key and shape checks, including abstract models."""
    state = nnx.state(model, WEIGHTS)
    pure = nnx.to_pure_dict(state)
    paths = jax.tree_util.tree_flatten_with_path(pure)[0]
    expected = {jax.tree_util.keystr(path) for path, _ in paths}
    if expected != set(weights):
        raise ValueError(
            f"Weight keys differ: missing {sorted(expected - set(weights))}, "
            f"extra {sorted(set(weights) - expected)}"
        )

    def replace(path: tuple, value: jax.Array) -> jax.Array:
        name = jax.tree_util.keystr(path)
        incoming = weights[name]
        if incoming.shape != value.shape:
            raise ValueError(
                f"Weight shape mismatch for {name}: {incoming.shape} != {value.shape}"
            )
        return jnp.asarray(incoming)

    nnx.replace_by_pure_dict(state, jax.tree_util.tree_map_with_path(replace, pure))
    nnx.update(model, state)


class PEACE(nnx.Module):
    """Embed wet audio and Faust effect code in a shared retrieval space.

    Use ``from_pretrained`` for trained weights. Direct construction initializes
    an untrained model and is primarily useful for testing.
    """

    def __init__(
        self,
        config: PEACEConfig | None = None,
        *,
        tokenizer: Tokenizer | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Build a model.

        Args:
            config: Architecture and effect library; defaults to the BoxGraph
                release architecture.
            tokenizer: SentencePiece tokenizer, required for a T5 code encoder.
            rngs: Parameter initialization keys.
        """
        self.config = config if config is not None else PEACEConfig()
        cfg = self.config
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.library = EffectLibrary(cfg.effects)
        self.audio_encoder = AudioEncoder(cfg.audio, rngs)
        self.audio_heads = Heads(
            self.audio_encoder.out_features, cfg.embed_dim, cfg.predictor_hidden_dim, rngs
        )
        if isinstance(cfg.code, T5Config):
            if tokenizer is None:
                raise ValueError("A T5 code encoder needs a tokenizer")
            self.code_encoder = T5CodeEncoder(
                cfg.code, self.library, CodeTokenizer(tokenizer, cfg.code), rngs
            )
        else:
            self.code_encoder = BoxGraphEncoder(cfg.code, self.library, rngs)
        self.code_heads = Heads(
            self.code_encoder.out_features, cfg.embed_dim, cfg.predictor_hidden_dim, rngs
        )
        self.eval()

    @property
    def sample_rate(self) -> int:
        """Native waveform sample rate."""
        return self.config.audio.sample_rate

    @nnx.jit
    def embed_audio(self, audio: AudioTree) -> Embeddings:
        """Return latents, projections, and predictions for wet audio.

        Audio is resampled to 48 kHz, made stereo, normalized to -18 LUFS, and
        divided by its peak when that exceeds one, as in training.

        Args:
            audio: AudioTree with waveform ``[batch, channels, samples]`` and one
                or two channels. At least 32 spectrogram frames (0.65 s) are
                required; the model was trained on 6-second excerpts.

        Returns:
            Embeddings with latents ``[batch, 1024]`` and unit-length
            projections and predictions ``[batch, embed_dim]``.
        """
        if audio.num_channels not in (1, 2):
            raise ValueError("PEACE expects mono or stereo audio")
        audio = audio.resample(self.sample_rate).to_stereo()
        audio = audio.normalize_lufs(self.config.audio.normalize_lufs)
        waveform = audio.waveform
        peak = jnp.max(jnp.abs(waveform), axis=(-2, -1), keepdims=True)
        waveform = waveform / jnp.maximum(peak, 1.0)
        return self.audio_heads(self.audio_encoder(waveform))

    def embed_code(
        self,
        programs: str | Sequence[str],
        mask_parameters: bool = False,
        batch_size: int = 256,
    ) -> Embeddings:
        """Return latents, projections, and predictions for Faust programs.

        Args:
            programs: Faust source whose ``process`` chains library effects
                with ``:``; see ``EffectLibrary.parse``. The T5 encoder reads
                the text, so write programs as ``EffectLibrary.program`` does.
            mask_parameters: Mask every parameter value so embeddings describe
                only which effects are used and in what order.
            batch_size: Programs encoded per compiled call.

        Returns:
            Embeddings with latents ``[programs, 512]`` and unit-length
            projections and predictions ``[programs, embed_dim]``.
        """
        if isinstance(programs, str):
            programs = [programs]
        if not programs:
            raise ValueError("No programs to encode")
        outputs = []
        for start in range(0, len(programs), batch_size):
            inputs, options = self.code_encoder.prepare(
                programs[start : start + batch_size], mask_parameters
            )
            outputs.append(self._embed_code(inputs, options))
        return Embeddings(*(jnp.concatenate(parts) for parts in zip(*outputs)))

    @nnx.jit(static_argnames="options")
    def _embed_code(self, inputs, options: tuple) -> Embeddings:
        """Encode inputs from ``code_encoder.prepare`` and apply the code heads."""
        return self.code_heads(self.code_encoder(inputs, *options))

    def encode_audio(self, audio: AudioTree) -> jax.Array:
        """Return unit-length audio retrieval embeddings ``[batch, embed_dim]``."""
        return self.embed_audio(audio).prediction

    def encode_code(
        self, programs: str | Sequence[str], mask_parameters: bool = False
    ) -> jax.Array:
        """Return unit-length code retrieval embeddings ``[programs, embed_dim]``."""
        return self.embed_code(programs, mask_parameters=mask_parameters).prediction

    def save_pretrained(self, directory: str | Path) -> None:
        """Save architecture, effect library, weights, and any tokenizer locally."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        config = {"format_version": FORMAT_VERSION, **asdict(self.config)}
        encoder = next(k for k, v in CODE_ENCODERS.items() if isinstance(self.config.code, v))
        config["code"] = {"encoder": encoder, **config["code"]}
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        save_file(flat_weights(self), str(destination / "model.safetensors"))
        if isinstance(self.code_encoder, T5CodeEncoder):
            self.code_encoder.tokenizer.save(destination / TOKENIZER_FILE)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        *,
        subfolder: str = "",
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> "PEACE":
        """Load an exported directory or a Hugging Face model repository.

        Args:
            path_or_repo: Local directory or Hub repository identifier.
            subfolder: Optional variant directory inside the export or repository.
            revision: Optional Hub revision or commit hash.
            local_files_only: Require already-cached Hub files when True.

        Returns:
            A fully restored model in evaluation mode.
        """
        location = Path(path_or_repo).expanduser()
        if not location.is_dir():
            from huggingface_hub import snapshot_download

            prefix = f"{subfolder}/" if subfolder else ""
            location = Path(
                snapshot_download(
                    str(path_or_repo),
                    revision=revision,
                    local_files_only=local_files_only,
                    allow_patterns=[
                        prefix + name
                        for name in ("config.json", "model.safetensors", TOKENIZER_FILE)
                    ],
                )
            )
        location = location / subfolder
        config = json.loads((location / "config.json").read_text())
        if config.pop("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported PEACE export format")
        audio = config.pop("audio")
        audio["channels"] = tuple(audio["channels"])
        code = config.pop("code")
        code_config = CODE_ENCODERS[code.pop("encoder")](**code)
        effects = parse_effects(config.pop("effects"))
        config = PEACEConfig(
            audio=AudioEncoderConfig(**audio),
            code=code_config,
            effects=effects,
            **config,
        )
        tokenizer = None
        if isinstance(code_config, T5Config):
            tokenizer = Tokenizer.from_file(str(location / TOKENIZER_FILE))
        model = nnx.eval_shape(lambda: cls(config, tokenizer=tokenizer))
        load_weights(model, load_file(str(location / "model.safetensors")))
        model.eval()
        return model
