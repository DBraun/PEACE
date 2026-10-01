"""Tests for the shipped Faust effect library and its renderer."""

import numpy as np
import pytest
from audiotree import AudioTree

from peace import PEACE, PEACEConfig
from peace.boxgraph import BoxGraphConfig
from peace.faust import dsp_dir, dsp_path, faust_source
from peace.graph import EffectLibrary
from peace.model import default_effects

LIBRARY = EffectLibrary(default_effects())
DELAY = "delay(1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)"
FREEVERB = "reverb_freeverb(0.1, 0.2, 0.3, 0.4)"


def test_every_effect_ships_its_faust_source() -> None:
    assert sorted(p.stem for p in dsp_dir().glob("*.dsp")) == sorted(
        e.function for e in LIBRARY.effects
    )


def test_chain_lists_effects_in_processing_order() -> None:
    program = f"box1 = {FREEVERB};\nprocess = {DELAY} : box1 : stereo_width(0.25);"
    chain = LIBRARY.chain(program)
    assert [e.function for e, _ in chain] == ["delay", "reverb_freeverb", "stereo_width"]
    assert chain[0][1] == (1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    assert chain[2][1] == (0.25,)


def test_faust_source_binds_each_effect_to_its_file() -> None:
    source = faust_source(f"process = {DELAY};", LIBRARY)
    assert f'delay = component("{dsp_path(LIBRARY.by_function["delay"]).as_posix()}");' in source
    assert source.endswith(f"process = {DELAY};")
    with pytest.raises(ValueError, match="Unknown effect"):
        faust_source("process = missing(0.5);", LIBRARY)


@pytest.fixture(scope="module")
def renderer():
    pytest.importorskip("dawdreamer")
    from peace.faust import Renderer

    return Renderer(LIBRARY)


def test_renderer_compiles_effects_and_keeps_shape(renderer) -> None:
    """Every effect but klonCentaur, whose compile alone takes minutes."""
    rng = np.random.default_rng(0)
    dry = AudioTree.create((0.1 * rng.standard_normal((2, 1, 9600))).astype(np.float32), 48000)
    for effect in LIBRARY.effects:
        if effect.function == "klonCentaur":
            continue
        values = ", ".join(str(p.default) for p in effect.parameters)
        wet = renderer(f"process = {effect.function}({values});", dry)
        assert wet.waveform.shape == (2, 2, 9600), effect.function
        assert np.isfinite(wet.waveform).all(), effect.function
    chained = renderer(f"process = {DELAY} : {FREEVERB};", dry)
    assert chained.waveform.shape == (2, 2, 9600)


def test_rendered_programs_embed_with_the_model(renderer) -> None:
    """A rendered program and its source feed the two encoders."""
    model = PEACE(
        PEACEConfig(code=BoxGraphConfig(hidden_dim=16, out_dim=8), embed_dim=8, predictor_hidden_dim=16)
    )
    program = f"process = {DELAY};"
    wet = renderer(program, AudioTree.create(np.zeros((1, 2, 48000), np.float32), 48000))
    assert model.encode_code(program).shape == (1, 8)
    assert wet.waveform.shape == (1, 2, 48000)
