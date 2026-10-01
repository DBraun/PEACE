import("stdfaust.lib");

reverse_echo_demo(delayL, delayR, wet) = dryWetMixerConstantPower(wet, (ef.reverseEchoN(1,delMaxL), ef.reverseEchoN(1,delMaxR)))
with {
    toSafeMax = aa.clip(0, 4) : 2^int(_+14); // delay line length
    delMaxL = toSafeMax(delayL);
    delMaxR = toSafeMax(delayR);
};

process = reverse_echo_demo;
// process = reverse_echo_demo(0, 3, 0.5);

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
