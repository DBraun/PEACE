// Klon Centaur overdrive for PEACE: ve.klonCentaur from the Faust libraries
// (vaeffects.lib, a port of Chowdhury-DSP ChowCentaur, BSD 3-Clause) on each
// stereo channel. Parameters (gain, treble, level) arrive as signal inputs.
import("stdfaust.lib");

process(gain, treble, level) =
    ve.klonCentaur(gain, treble, level), ve.klonCentaur(gain, treble, level);
