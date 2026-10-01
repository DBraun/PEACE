import("stdfaust.lib");
process = dattorro_rev_demo;

dattorro_rev_demo(in_bw, in_diffusion1, in_diffusion2, d_diff1, d_diff2, decay, damping, wet) =  dryWetMixerConstantPower(wet, re.dattorro_rev(pre_delay, in_bw, in_diffusion1, in_diffusion2, decay, d_diff1, d_diff2, damping))
with {
    pre_delay = 0;
};

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
