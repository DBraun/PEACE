"""Compile and render PEACE's Faust effect programs.

The effect library ships as Faust source in ``peace/dsp``. ``faust_source`` turns
a PEACE program into a self-contained Faust program for any Faust toolchain, and
``Renderer`` applies a program to audio with DawDreamer (``pip install
peace[render]``), rendering as the training data was.
"""

from importlib import resources
from pathlib import Path
from typing import Sequence

import numpy as np
from audiotree import AudioTree

from .graph import Effect, EffectLibrary

# Training prepended this much silence before rendering and trimmed it after, so
# effects with onset transients (KlonCentaur clicks) start settled.
WARMUP_SECONDS = 0.1


def dsp_dir() -> Path:
    """Directory holding the effect library's Faust source files."""
    return Path(str(resources.files(__package__).joinpath("dsp")))


def dsp_path(effect: Effect) -> Path:
    """Faust source file implementing ``effect``."""
    return dsp_dir() / f"{effect.function}.dsp"


def faust_source(program: str, library: EffectLibrary) -> str:
    """A self-contained Faust program for a PEACE program.

    Each library effect becomes ``name = component("<path>");``. An effect's
    parameters are its first inputs, so a call such as ``delay(0, 0.3, ...)``
    fixes them and leaves a stereo effect. Pass ``dsp_dir()`` to the Faust
    compiler's import path (``-I``) if the paths are not absolute there.
    """
    library.parse(program)  # reject programs the model cannot embed
    components = [
        f'{e.function} = component("{dsp_path(e).as_posix()}");' for e in library.effects
    ]
    return "\n".join(['import("stdfaust.lib");', *components, "", program])


class _CompiledEffect:
    """A DawDreamer engine, playback source, and Faust processor for one effect.

    Each effect gets its own engine because DawDreamer cannot add processors to
    an engine that has already rendered; this lets effects compile on first use.
    """

    def __init__(
        self, effect: Effect, sample_rate: int, buffer_size: int, search_paths: list[str]
    ) -> None:
        import dawdreamer

        self.sample_rate = sample_rate
        self.engine = dawdreamer.RenderEngine(sample_rate, buffer_size)
        self.playback = self.engine.make_playback_processor(
            "playback", np.zeros((2, 1), np.float32)
        )
        self.processor = self.engine.make_faust_processor(effect.function)
        self.processor.faust_libraries_paths = search_paths
        self.processor.set_dsp_string(dsp_path(effect).read_text())
        if not self.processor.compile():
            raise RuntimeError(f"Faust failed to compile {dsp_path(effect)}")
        inputs = len(effect.parameters) + 2
        if self.processor.get_num_input_channels() != inputs:
            raise RuntimeError(f"{effect.function} does not take {inputs} inputs")
        self.engine.load_graph([(self.playback, []), (self.processor, ["playback"])])

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Process parameter rows followed by stereo audio ``[params + 2, T]``."""
        self.playback.set_data(signal)
        length = signal.shape[1]
        self.engine.render(length / self.sample_rate)
        return self.engine.get_audio()[:, :length].astype(np.float32)


class Renderer:
    """Apply PEACE programs to stereo audio with DawDreamer.

    A chain renders effect by effect, with parameters as constant input signals
    ahead of the stereo audio, as the training data was rendered. Effects compile
    from ``peace/dsp`` on first use and are then reused. Most compile in under a
    second; ``klonCentaur`` takes a few minutes.

    Args:
        library: Effects the programs may use, typically ``model.library``.
        sample_rate: Rendering sample rate.
        faust_libraries: Extra Faust ``.lib`` directories searched first, such as
            a current ``faustlibraries`` checkout.
        buffer_size: DawDreamer block size.
    """

    def __init__(
        self,
        library: EffectLibrary,
        sample_rate: int = 48_000,
        faust_libraries: Sequence[str | Path] = (),
        buffer_size: int = 512,
    ) -> None:
        import dawdreamer  # noqa: F401  (fail here, not mid-render, if missing)

        self.library = library
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.search_paths = [*map(str, faust_libraries), str(dsp_dir())]
        self._compiled: dict[str, _CompiledEffect] = {}

    def _effect(self, effect: Effect) -> _CompiledEffect:
        if effect.function not in self._compiled:
            self._compiled[effect.function] = _CompiledEffect(
                effect, self.sample_rate, self.buffer_size, self.search_paths
            )
        return self._compiled[effect.function]

    def __call__(self, program: str, audio: AudioTree) -> AudioTree:
        """Render ``program`` on each item of ``audio``.

        Args:
            program: PEACE program; see ``EffectLibrary.parse``.
            audio: Dry audio ``[batch, channels, samples]``, mono or stereo.

        Returns:
            Wet stereo audio with the input's shape at ``sample_rate``.
        """
        chain = [(self._effect(e), values) for e, values in self.library.chain(program)]
        audio = audio.resample(self.sample_rate).to_stereo()
        dry = np.asarray(audio.waveform, np.float32)
        warmup = int(WARMUP_SECONDS * self.sample_rate)
        return AudioTree.create(
            np.stack([self._render(chain, item, warmup) for item in dry]),
            self.sample_rate,
        )

    @staticmethod
    def _render(chain, item: np.ndarray, warmup: int) -> np.ndarray:
        signal = np.concatenate([np.zeros((2, warmup), np.float32), item], axis=1)
        for compiled, values in chain:
            params = np.repeat(np.asarray(values, np.float32)[:, None], signal.shape[1], 1)
            signal = compiled(np.concatenate([params, signal]))
        return signal[:, warmup:]
