import("stdfaust.lib");

// grainSize = hslider("Grain Size[unit:samp]", 2048, 128, 4096, 1);
// pitch = hslider("Pitch [unit:semi]", 0, -24, 24, .5);

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

pitchShift(_grainSize, _pitch, _mix) = dryWetMixerConstantPower(mix, sp.stereoize(ef.transpose(grainSize, grainSize*.5, pitch)))
with {
    grainSize = _grainSize : it.remap(0, 1, 128, 4096);
    pitch = _pitch : it.remap(0, 1, -24, 24);
    mix = _mix;
};

pitchShiftUI = pitchShift(grainSize, pitch, mix) with {
    grainSize = hslider("Grain Size", 0.5, 0, 1, .01);
    pitch = hslider("Pitch", 0.5, 0, 1, .01) ;
    mix = hslider("Mix", 1, 0, 1, .01);
};

process = pitchShift;
// process = pitchShiftUI;
