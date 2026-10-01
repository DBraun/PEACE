"""AFx-Rep audio encoder: a mid/side CNN14 over log-mel spectrograms.

The architecture follows AFx-Rep (Steinmetz et al., 2024), whose backbone is the
PANNs CNN14 (Kong et al., 2020). Mid ``(L + R) / 2`` and side ``(L - R) / 2``
spectrograms share the convolutional stack, and separate linear heads produce
512-dimensional mid and side embeddings.
"""

from dataclasses import dataclass

import jax
import librosax
import librosax.feature  # noqa: F401  (registers librosax.feature.melspectrogram)
from einops import rearrange
from flax import nnx
from jax import numpy as jnp



@dataclass
class AudioEncoderConfig:
    """Spectrogram frontend and CNN14 configuration.

    Attributes:
        sample_rate: Waveform sample rate (Hz).
        n_fft: FFT and window size.
        hop_length: STFT hop in samples.
        n_mels: Number of mel bands.
        f_min: Lowest mel frequency (Hz).
        f_max: Highest mel frequency (Hz).
        channels: Output width of each convolutional block.
        embed_dim: Mid and side embedding width.
        normalize_lufs: Loudness target applied to every input.
    """

    sample_rate: int = 48_000
    n_fft: int = 2048
    hop_length: int = 1024
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 20000.0
    channels: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
    embed_dim: int = 512
    normalize_lufs: float = -18.0


def l2_normalize(x: jax.Array) -> jax.Array:
    """Scale the last axis to unit length, as in training."""
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class ConvBlock(nnx.Module):
    """Two 3x3 convolutions with batch norm and ReLU, then average pooling."""

    def __init__(self, in_channels: int, out_channels: int, pool: int, rngs: nnx.Rngs):
        conv = dict(kernel_size=(3, 3), padding=((1, 1), (1, 1)), use_bias=False)
        self.conv1 = nnx.Conv(in_channels, out_channels, **conv, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, **conv, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.pool = pool

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = nnx.relu(self.bn2(self.conv2(x)))
        return nnx.avg_pool(x, (self.pool, self.pool), strides=(self.pool, self.pool))


class AudioEncoder(nnx.Module):
    """Map stereo waveforms to concatenated unit-length mid and side embeddings."""

    def __init__(self, config: AudioEncoderConfig, rngs: nnx.Rngs) -> None:
        self.config = config
        features = 2 * config.n_mels
        self.input_mean = nnx.BatchStat(jnp.zeros(features))
        self.input_std = nnx.BatchStat(jnp.ones(features))
        widths = (1, *config.channels)
        last = len(config.channels) - 1
        self.blocks = nnx.List(
            [
                ConvBlock(widths[i], widths[i + 1], 1 if i == last else 2, rngs)
                for i in range(len(config.channels))
            ]
        )
        self.fc_mid = nnx.Linear(widths[-1], config.embed_dim, rngs=rngs)
        self.fc_side = nnx.Linear(widths[-1], config.embed_dim, rngs=rngs)

    @property
    def out_features(self) -> int:
        return 2 * self.config.embed_dim

    def mel(self, waveform: jax.Array) -> jax.Array:
        """Compute training-time log-mel features scaled to ``[-1, 1]``.

        Args:
            waveform: Stereo audio ``[batch, 2, samples]`` at the native rate.

        Returns:
            Mid and side features ``[batch, 2, frames, n_mels]``.
        """
        cfg = self.config
        mid = (waveform[:, 0] + waveform[:, 1]) / 2
        side = (waveform[:, 0] - waveform[:, 1]) / 2
        power = librosax.feature.melspectrogram(
            y=jnp.stack([mid, side], axis=1),
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            power=2.0,
        )
        db = librosax.power_to_db(power, ref=1.0, amin=1e-10, top_db=None)
        db = jnp.clip(rearrange(db, "b c f t -> b c t f"), -80.0, 40.0)
        return (db + 80.0) / 60.0 - 1.0

    def __call__(self, waveform: jax.Array) -> jax.Array:
        """Return ``[batch, 2 * embed_dim]`` mid and side embeddings.

        Args:
            waveform: Loudness-normalized stereo audio ``[batch, 2, samples]``.
        """
        min_frames = 2 ** (len(self.config.channels) - 1)
        if 1 + waveform.shape[-1] // self.config.hop_length < min_frames:
            min_samples = (min_frames - 1) * self.config.hop_length
            raise ValueError(f"Audio needs at least {min_samples} samples")
        return self.encode_mel(self.mel(waveform))

    def encode_mel(self, mel: jax.Array) -> jax.Array:
        """Return ``[batch, 2 * embed_dim]`` embeddings from ``mel`` features.

        Args:
            mel: Mid and side features ``[batch, 2, frames, n_mels]`` from ``mel``.
        """
        batch, channels, frames, bins = mel.shape
        x = rearrange(mel, "b c t f -> b t (c f)")
        x = (x - self.input_mean[...]) / self.input_std[...]
        x = rearrange(x, "b t (c f) -> (b c) t f 1", c=channels, f=bins)
        for block in self.blocks:
            x = block(x)
        x = jnp.mean(x, axis=2)
        x = jnp.max(x, axis=1) + jnp.mean(x, axis=1)
        x = rearrange(x, "(b c) d -> b c d", b=batch, c=channels)
        mid = l2_normalize(self.fc_mid(x[:, 0]))
        side = l2_normalize(self.fc_side(x[:, 1]))
        return jnp.concatenate([mid, side], axis=-1)
