// SPDX-License-Identifier: MIT
// Spring reverb for PEACE, adapted from re.springreverb by Daniel Leonov in the
// Faust libraries (https://github.com/grame-cncm/faustlibraries, reverbs.lib),
// whose demo declares the MIT license. Changes: one reverb per stereo channel,
// and a constant-power wet/dry mix in place of adding blend-scaled wet signal.
// Parameters are passed as signal inputs, like other effects in effects_library.yaml
import("stdfaust.lib");

// springreverb_stereo takes 6 parameter signals + 2 audio signals
// Parameters: Dwell, Tone, Tension, Springs, Wet
// Springs is an integer in [0, 1, 2] selecting delay spread preset
springreverb_stereo(dwell_aux, tone_aux, tension_aux, springs, wetmix) =
    dryWetMixerConstantPower(wetmix, (mono_fx, mono_fx))
with {
    SR = ma.SR;

    // Parameter remapping from external [0..1] to internal [0..10]
    dwell = dwell_aux * 10;
    tone = tone_aux * 10;
    tension = tension_aux * 10;

    clamp(x, lo, hi) = max(lo, min(hi, x));

    dwell_ctrl = clamp(dwell, 0, 10);
    tone_ctrl = clamp(tone, 0, 10);
    tension_ctrl = clamp(tension, 0, 10);
    springs_index = clamp(round(springs), 0, 2);

    // Quadratic fit from original dwell sweep measurements
    feedback_gain_linear = dwell_ctrl * (-0.0006 * dwell_ctrl + 0.013) + 0.26;

    // Tone affects lowpass cutoff on wet path
    lowpass_freq_hz = tone_ctrl * (190 * tone_ctrl + 50) + 1500;
    makeup_gain = 0.5 * min(5, 1 + 2000 / lowpass_freq_hz);

    // Tension affects base delay time
    base_spring_delay_s = tension_ctrl * (0.00032 * tension_ctrl - 0.0072) + 0.07;
    spread = spread_choice(springs_index);

    diffusion_delay_max_samples = 0.035 * SR : round;
    spring_delay_max_samples = 0.08 * SR : round;

    N = 8;

    diffusion_delay_samples(min_, max_, bin, bins) =
        abs(min_ + ((max_ - min_) / bins) * bin) * SR : round;

    diffusion =
        par(i, N, de.delay(diffusion_delay_samples(0.05, 0.020, i, 8), diffusion_delay_max_samples))
        : ro.hadamard(N)
        : polarity_flips_a
        : par(i, N, de.delay(diffusion_delay_samples(0.009, 0.030, i, N), diffusion_delay_max_samples))
        : ro.hadamard(N)
        : polarity_flips_b
        : par(i, N, de.delay(diffusion_delay_samples(0.010, 0.025, i, N), diffusion_delay_max_samples))
        : ro.hadamard(N)
        : polarity_flips_c
        : par(i, N, de.delay(diffusion_delay_samples(0.009, 0.032, i, N), diffusion_delay_max_samples))
    with {
        polarity_flips_a = _, *(-1), _, _, *(-1), _, *(-1), *(-1);
        polarity_flips_b = *(-1), _, _, *(-1), _, _, _, *(-1);
        polarity_flips_c = _, *(-1), *(-1), _, _, _, *(-1), _;
    };

    spring(d, lp) = de.delay(spring_delay_max_samples, d) : fi.lowpass(1, lp);

    spring_delay_samples(i) =
        (base_spring_delay_s + ba.take(i + 1, offsets) * spread) * SR : round
    with {
        offsets = 1, 2, 3, 5, 7, 11, 13, 17;
    };

    delay_lines = par(i, N, spring(spring_delay_samples(i), lowpass_freq_hz) : aa.hardclip);
    feedback_lines = ro.hadamard(N) : par(i, N, *(feedback_gain_linear));

    mono_fx = *(0.01) <: diffusion
        : (si.bus(N * 2) :> delay_lines) ~ (feedback_lines)
        :> fi.highpass(1, 150) : *(makeup_gain);

    spread_choice(idx) = ba.if(idx == 0, 0.2e-5, ba.if(idx == 1, 5.0e-4, 2.8e-5));
};

process = springreverb_stereo;
// process = springreverb_stereo(
//     hslider("dwell", 0.5, 0, 1, .01),
//     hslider("tone", 0.5, 0, 1, .01),
//     hslider("tension", 0.5, 0, 1, .01),
//     hslider("springs", 0.5, 0, 1, .01),
//     hslider("wet", 0.5, 0, 1, .01)
// );

// Implementation to share common code
dwmEnv(wetAmount, FX) = environment
{
    N = inputs(FX);
    wet(wg) = FX : par(i, N, *(wg));
    dry(dg) = par(i, N, *(dg));
    out(wg, dg) = si.bus(N) <: wet(wg), dry(dg) :> si.bus(N);

    dryWetMixer = out(wetGain, dryGain)
    with {
        wetGain = wetAmount;
        dryGain = 1. - wetGain;
    };

    dryWetMixerConstantPower = out(wetGain, dryGain)
    with {
        theta = ma.PI*wetAmount/2.;
        dryGain = cos(theta);
        wetGain = sin(theta);
    };
};

dryWetMixerConstantPower(wetAmount, FX) = dwmEnv(wetAmount, FX).dryWetMixerConstantPower;
