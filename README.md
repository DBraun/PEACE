# PEACE

This standalone JAX/Flax package accompanies **PEACE: Joint Embeddings of DSP Effects Code and Audio**, by David Braun and Adam Finkelstein ([ISMIR 2026](https://ismir2026.ismir.net/)) ([arxiv](https://arxiv.org/abs/2610.03405)). It's a joint embedding of effects programs written in the Faust programming language and their corresponding "wet"/"processed"/"effected" audio. PEACE is an acronym of **P**airwise **E**ffected **A**udio **C**ode **E**mbeddings.

Try the embedding [explorer](https://dbraun.github.io/PEACE-map/) and [chain retrieval demo](https://dirt.design/PEACE-map/chain-retrieval/).

With PEACE you can:

- **Retrieve effect code from audio**: rank a gallery of Faust programs by how well each explains a wet recording.
- **Retrieve audio from code**: rank recordings by how well they match a program.
- **Identify an effect chain without its settings**: mask every parameter value so a program's embedding describes only which effects it uses and in what order.
- **Apply what you find**: render any retrieved program on new audio with the bundled Faust sources of the 21 effects.
- **Compare audio with audio**: the audio embeddings capture effect character, such as a reverb's room, largely independently of the source content (i.e., whether it's guitar or singing)

## Install

```bash
pip install "peace @ git+https://github.com/DBraun/PEACE"
```

For a GPU, install the JAX accelerator build for your system.
Programs are parsed with [tree-sitter-faust](https://pypi.org/project/tree-sitter-faust/), so embedding needs no Faust compiler.
To render programs, add [DawDreamer](https://github.com/DBraun/DawDreamer) with the `render` extra:

```bash
pip install "peace[render] @ git+https://github.com/DBraun/PEACE"
```

## Choose a model

Two models are released on [Hugging Face](https://huggingface.co/davidbraun/peace).
Both use the same audio encoder architecture ([AFx-Rep](https://github.com/csteinmetz1/st-ito)'s CNN) and retrieve chains of one to three effects about equally well.
The models differ in how they encode the Faust programs.

- `boxgraph` parses each program into the graph resembling an [intermediate stage](https://faustdoc.grame.fr/tutorials/box-api/) of the Faust compiler. It parses parameter values with the precision of an ordinary compiler. Formatting, comments, and how the program is written do not affect the embeddings.
- `t5` tokenizes the program's source text, with decimals rounded to two places. It holds up better on chains longer than the three effects seen in training, but it should be given programs in the layout shown a few paragraphs below.

```python
from peace import PEACE

model = PEACE.from_pretrained("davidbraun/peace", subfolder="boxgraph")  # or "t5"
```

## Retrieve effect code from audio

```python
import numpy as np
from audiotree import AudioTree

programs = [
    model.library.program([("reverb_freeverb", [0.8, 0.5, 0.3, 0.4])]),
    model.library.program([
        ("distortion", [1, 3, 0.6, 0.5, 0.2, 0.7, 1.0]),
        ("delay", [0, 0.3, 0.5, 0.2, 0.5, 0.2, 0.5, 0.5, 0.4]),
    ]),
]
gallery = np.asarray(model.encode_code(programs))  # [2, 768]

wet = AudioTree.from_file("wet.wav", duration=6)
query = np.asarray(model.encode_audio(wet))  # [1, 768]
ranking = np.argsort(-(query @ gallery.T), axis=-1)
```

Both encoders return unit-length embeddings, so a matrix product gives cosine similarity in either direction;
`gallery @ query.T` ranks audio for each program.
Audio carries its sample rate in [AudioTree](http://dbraun.github.io/audiotree/introduction/introduction.html), and any rate, mono or stereo, is accepted:
the model resamples to 48 kHz, levels each clip to −18 LUFS, and limits its peak, as during training.
Clips must be at least 0.66 seconds long, and the model was trained on 6-second excerpts.
`encode_code` encodes large galleries in chunks.

## Write programs

A program defines `process` as library effects joined by `:`.
Each effect is called with every one of its parameters, in library order:

```faust
box1 = filter(
    10,  // mode
    0.86,  // cutoff
    0.34,  // res
    0.67,  // drive
    0.68,  // other
    0.12  // mix
);

box2 = reverb_freeverb(
    0.80,  // damp
    0.50,  // roomSize
    0.30,  // stereoSpread
    0.40  // mix
);

process = box1 : box2;
```

This is the layout of the training programs, and `model.library.program` writes it from `(effect, values)` pairs. `model.library.chain` reads a program back into those pairs.
The `BoxGraph` model also accepts effects written inline, such as `process = stereo_width(0.25) : reverb_freeverb(0.8, 0.5, 0.3, 0.4);`, but the T5 model reads the text itself, so give it programs with comments in the layout above.

Float parameters lie in `[0, 1]`;
integer parameters select one of the parameter's `num_values` modes.
The effects and their parameters are listed in `model.library.effects` and in [`effects.json`](src/peace/effects.json).
Programs the models were not trained on raise `ValueError`: unknown effects, wrong argument counts, out-of-range values, and composition other than `:`, such as `,`, `<:`, `:>`, or `~`.

Chains may be longer than the three effects seen in training;
the `BoxGraph` model runs one message-passing step per graph level so information reaches every effect.
The `T5` model reads at most 588 tokens.
An effect takes 18 to 180 tokens in the layout above, depending on its number of parameters, so very long chains are truncated.

## Identify an effect chain

```python
chains = [
    model.library.program([("reverb_zita", [0.5] * 10), ("delay", [0] + [0.5] * 8)]),
    model.library.program([("delay", [0] + [0.5] * 8), ("reverb_zita", [0.5] * 10)]),
]
topologies = np.asarray(model.encode_code(chains, mask_parameters=True))
best = np.argmax(query @ topologies.T, axis=-1)
```

With `mask_parameters=True`, embeddings ignore parameter values and describe only which effects a program uses and in what order, so placeholder values are fine.
Embed every candidate chain this way and compare them with audio to identify the chain that processed a recording without estimating its settings.
`BoxGraph` replaces each value with a learned mask token.
`T5` hides the tokens of every number, which still occupy positions, so keep the number of digits fixed, as `library.program` does.

## Render effect programs

```python
from peace.faust import Renderer, faust_source

renderer = Renderer(model.library)
dry = AudioTree.from_file("dry.wav", sample_rate=48_000)
wet = renderer(programs[1], dry)  # AudioTree [1, 2, samples]
print(faust_source(programs[1], model.library))
```

`Renderer` applies a chain effect by effect with DawDreamer, as the training audio was rendered. Effects compile on first use;
`klonCentaur` takes a few minutes.
`faust_source` returns a self-contained Faust program that loads each effect from its file in [`src/peace/dsp`](src/peace/dsp), for use with any Faust toolchain.
The `render` extra pins DawDreamer 0.9.0, whose bundled [Faust libraries](https://github.com/grame-cncm/faustlibraries) compile every effect;
`faust_libraries=` adds directories searched before them.

## Use the embeddings directly

```python
audio = model.embed_audio(wet)     # latent [1, 1024]; projection and prediction [1, 768]
code = model.embed_code(programs)  # latent [2, 512]; projection and prediction [2, 768]
```

`embed_audio` and `embed_code` return the encoder `latent`, the unit-length `projection`, and the unit-length `prediction` used for retrieval.
For audio-to-audio comparisons such as matching reverbs, the audio `prediction` works better than the `latent`.

Models save and load as local directories, or from any Hugging Face repository with `subfolder` and `revision`:

```python
model.save_pretrained("models/boxgraph")
restored = PEACE.from_pretrained("models/boxgraph")
```

## Limitations

- The models know 21 effects, applied in series. Programs with other effects or with parallel or feedback routing are rejected rather than approximated.
- Some settings cannot be told apart from audio alone. Swapping two linear time-invariant effects in a chain does not change the output, a limiter does not change quiet audio, a gate does not change loud audio, and stereo width does nothing to mono audio.
- The audio features cannot tell left from right, so a pan and its mirror image embed identically.
- The training audio used randomly sampled parameters filtered for loudness rather than curated presets.

## Audio features and published versions

PEACE's audio encoder shares AFx-Rep's architecture but not its exact input features: PEACE's log-mel frames are zero-padded at the ends of each clip, where AFx-Rep pads by reflection.
The two differ only in a clip's first and last frame.
This package computes PEACE's own features.
The ISMIR 2026 proceedings computed PEACE's features with reflection padding for two evaluations, RIR retrieval and chain retrieval;
the [arXiv version](https://arxiv.org/abs/2610.03405) recomputes them with each model's training features, so those numbers differ slightly between the two versions.

## Citation

```bibtex
@inproceedings{braun_peace_2026,
  title     = {{PEACE}: Joint Embeddings of {DSP} Effects Code and Audio},
  author    = {Braun, David and Finkelstein, Adam},
  booktitle = {Proc. of the 27th Int. Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2026},
  month     = nov,
}
```

## License

Code is MIT-licensed.
The released weights are CC BY-NC 4.0: non-commercial use with attribution.
The T5 weights are fine-tuned from [google/t5-v1_1-small](https://huggingface.co/google/t5-v1_1-small) (Apache License 2.0).
See [`NOTICE`](NOTICE) for the AFx-Rep, PANNs, and T5 attributions and for the terms and origins of the Faust effect sources.
