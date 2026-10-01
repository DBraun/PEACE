"""Tests for the standalone inference, parsing, and weight-loading contract."""

from pathlib import Path

from jax import numpy as jnp
import numpy as np
import pytest
from audiotree import AudioTree

from peace import PEACE, PEACEConfig
from peace.audio import AudioEncoderConfig
from peace.boxgraph import BoxGraphConfig
from peace.graph import NodeType
from peace.model import flat_weights, load_weights

CHAIN = (
    "box1 = delay(1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);  // Mode is discrete\n"
    "box2 = reverb_freeverb(0.1, 0.2, 0.3, 0.4);\n"
    "process = box1 : box2;"
)
SINGLE = "process = stereo_width(0.25);"


def call(effect, values=None) -> str:
    values = [p.default for p in effect.parameters] if values is None else values
    return f"{effect.function}({', '.join(str(v) for v in values)})"


@pytest.fixture(scope="module")
def model() -> PEACE:
    """A small instance of the actual architecture, without pretrained downloads."""
    return PEACE(
        PEACEConfig(
            audio=AudioEncoderConfig(channels=(4, 4, 4, 4, 4, 8), embed_dim=8),
            code=BoxGraphConfig(hidden_dim=16, out_dim=8),
            embed_dim=8,
            predictor_hidden_dim=16,
        )
    )


def stereo(seconds: float = 1.0, seed: int = 0) -> AudioTree:
    rng = np.random.default_rng(seed)
    waveform = 0.1 * rng.standard_normal((1, 2, int(48000 * seconds)))
    return AudioTree.create(jnp.asarray(waveform, jnp.float32), 48000)


def test_local_round_trip(model: PEACE, tmp_path: Path) -> None:
    """All inference weights and numerical outputs survive local export/reload."""
    model.save_pretrained(tmp_path)
    restored = PEACE.from_pretrained(tmp_path)
    original = flat_weights(model)
    actual = flat_weights(restored)
    assert original.keys() == actual.keys()
    for key in original:
        np.testing.assert_array_equal(original[key], actual[key])
    audio = stereo()
    np.testing.assert_allclose(restored.encode_audio(audio), model.encode_audio(audio), atol=1e-6)
    np.testing.assert_allclose(
        restored.encode_code([CHAIN, SINGLE]), model.encode_code([CHAIN, SINGLE]), atol=1e-6
    )


def test_weight_key_mismatch_fails(model: PEACE) -> None:
    """A partially downloaded or wrong model cannot silently initialize weights."""
    weights = flat_weights(model)
    weights.pop(next(iter(weights)))
    with pytest.raises(ValueError, match="Weight keys differ"):
        load_weights(model, weights)


def test_weight_shape_mismatch_fails(model: PEACE) -> None:
    """Architecture mismatches are rejected before inference."""
    weights = flat_weights(model)
    weights[next(iter(weights))] = np.zeros((1,))
    with pytest.raises(ValueError, match="Weight shape mismatch"):
        load_weights(model, weights)


def test_outputs_are_normalized(model: PEACE) -> None:
    """Projections and predictions from both arms have unit length."""
    for embeddings in (model.embed_audio(stereo()), model.embed_code([CHAIN, SINGLE])):
        for values in (embeddings.projection, embeddings.prediction):
            np.testing.assert_allclose(np.linalg.norm(values, axis=-1), 1, atol=1e-6)


def test_mono_matches_duplicated_stereo(model: PEACE) -> None:
    """Mono input is treated as a centered stereo signal."""
    mono = AudioTree.create(stereo().waveform[:, :1], 48000)
    doubled = AudioTree.create(jnp.repeat(mono.waveform, 2, axis=1), 48000)
    np.testing.assert_allclose(model.encode_audio(mono), model.encode_audio(doubled), atol=1e-6)


def test_audio_level_is_normalized(model: PEACE) -> None:
    """Loudness normalization removes gain as a cue."""
    audio = stereo()
    quieter = audio.replace(waveform=0.25 * audio.waveform)
    np.testing.assert_allclose(model.encode_audio(quieter), model.encode_audio(audio), atol=1e-5)


def test_short_audio_is_rejected(model: PEACE) -> None:
    """Clips shorter than the CNN's pooling depth fail clearly."""
    with pytest.raises(ValueError, match="at least"):
        model.encode_audio(stereo(0.3))


def test_parse_builds_boxgraph(model: PEACE) -> None:
    """An effect call is (params, bus) : effect, and chains nest seq nodes."""
    graph = model.library.parse(SINGLE)
    assert graph.node_types == [NodeType.FLOAT_PARAM, NodeType.BUS, NodeType.PAR, NodeType.DSP, NodeType.SEQ]
    assert sorted(graph.edges) == [(0, 2, 0), (1, 2, 1), (2, 4, 0), (3, 4, 1)]
    assert graph.depth == 2
    chain = model.library.parse(CHAIN)
    assert chain.node_types.count(NodeType.SEQ) == 3
    assert chain.node_types.count(NodeType.INT_PARAM) == 1
    assert chain.depth == 3


@pytest.mark.parametrize(
    ("program", "message"),
    [
        ("process = missing(0.5);", "Unknown effect"),
        ("process = stereo_width(0.2, 0.3);", "takes 1 arguments"),
        ("process = stereo_width(1.5);", r"must be in \[0, 1\]"),
        ("process = delay(3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);", "must be an integer"),
        ("process = stereo_width(0.2) , stereo_width(0.3);", "Unsupported"),
        ("box = stereo_width(0.2);", "no process"),
    ],
)
def test_parse_rejects_unsupported_programs(model: PEACE, program: str, message: str) -> None:
    """Programs outside the training distribution are rejected, not approximated."""
    with pytest.raises(ValueError, match=message):
        model.encode_code(program)


def test_batching_and_extra_layers_do_not_change_embeddings(model: PEACE) -> None:
    """Packing graphs together and running past the graph depth are both exact."""
    alone = model.encode_code(CHAIN)
    together = model.encode_code([SINGLE, CHAIN, SINGLE])
    np.testing.assert_allclose(together[1:2], alone, atol=1e-6)
    np.testing.assert_allclose(together[0], together[2], atol=1e-6)
    graphs = model.library.batch([model.library.parse(CHAIN)])
    deep = model._embed_code(graphs, (1, 9, False)).prediction
    np.testing.assert_allclose(deep, alone, atol=1e-6)


def test_long_chains_run_to_their_depth(model: PEACE) -> None:
    """Chains longer than training use one message-passing step per level."""
    effects = model.library.effects
    program = "process = " + " : ".join(call(e) for e in effects[:8]) + ";"
    assert model.library.parse(program).depth == 9
    graphs = model.library.batch([model.library.parse(program)])
    shallow, full, deeper = (
        model._embed_code(graphs, (1, layers, False)).latent for layers in (4, 9, 12)
    )
    np.testing.assert_allclose(model.embed_code(program).latent, full, atol=1e-6)
    np.testing.assert_allclose(deeper, full, atol=1e-6)
    assert np.abs(shallow - full).max() > 1e-5


def test_mask_parameters_keeps_only_effect_choice(model: PEACE) -> None:
    """Masked embeddings ignore parameter values but not which effects are used."""
    delay, freeverb, zita = (
        model.library.by_function[n] for n in ("delay", "reverb_freeverb", "reverb_zita")
    )
    first = f"process = {call(delay)} : {call(freeverb)};"
    other = f"process = {call(delay, [2] + [0.9] * 8)} : {call(freeverb, [0.0] * 4)};"
    different = f"process = {call(delay)} : {call(zita)};"
    masked = model.embed_code([first, other, different], mask_parameters=True).latent
    np.testing.assert_allclose(masked[0], masked[1], atol=1e-6)
    assert np.abs(masked[0] - masked[2]).max() > 1e-3
    visible = model.embed_code([first, other]).latent
    assert np.abs(visible[0] - visible[1]).max() > 1e-3


def test_chunked_encoding_matches_one_batch(model: PEACE) -> None:
    """Gallery chunks give the same embeddings as a single call."""
    programs = [f"process = {call(e)};" for e in model.library.effects]
    full = model.embed_code(programs, batch_size=len(programs)).prediction
    chunked = model.embed_code(programs, batch_size=4).prediction
    np.testing.assert_allclose(chunked, full, atol=1e-6)
    assert full.shape == (len(programs), 8)
