import("stdfaust.lib");

hzScale(lowHz, highHz) = it.remap(0, 1, ba.hz2midikey(lowHz), ba.hz2midikey(highHz)) : ba.midikey2hz;

declare zita_rev1 author "Julius O. Smith III";
declare zita_rev1 licence "MIT";

zita_rev1 = dryWetMixerConstantPower(drywet, (re.zita_rev1_stereo(rdel,f1,f2,t60dc,t60m,fsmax):out_eq))
with{
    fsmax = 48000.0;  // highest sampling rate that will be used

    fdn_group(x) = hgroup(
    "[0] Zita_Rev1 [tooltip: ~ ZITA REV1 FEEDBACK DELAY NETWORK (FDN) & SCHROEDER
    ALLPASS-COMB REVERBERATOR (8x8). See Faust's reverbs.lib for documentation and
    references]", x);

    in_group(x) = fdn_group(hgroup("[1] Input", x));

    // rdel = in_group(vslider("[1] In Delay [unit:ms] [style:knob] [tooltip: Delay in ms
    //     before reverberation begins]",60,20,100,1));
    rdel = in_group(vslider("[1] In Delay [style:knob] [tooltip: Delay in ms
        before reverberation begins]", 0.5, 0, 1, .01) : it.remap(0, 1, 20, 100));

    freq_group(x) = fdn_group(hgroup("[2] Decay Times in Bands (see tooltips)", x));

    // f1 = freq_group(vslider("[1] LF X [unit:Hz] [style:knob] [scale:log] [tooltip:
    //     Crossover frequency (Hz) separating low and middle frequencies]", 200, 50, 1000, 1));
    f1 = freq_group(vslider("[1] LF X [style:knob]
    [tooltip:Crossover frequency (Hz) separating low and middle frequencies]", 0.4, 0, 1, .01) : hzScale(50, 1000));

    // t60dc = freq_group(vslider("[2] Low RT60 [unit:s] [style:knob] [scale:log]
    // [style:knob] [tooltip: T60 = time (in seconds) to decay 60dB in low-frequency band]",
    // 3, 1, 8, 0.1));
        t60dc = freq_group(vslider("[2] Low RT60 [style:knob]
    [style:knob] [tooltip: T60 = time (in seconds) to decay 60dB in low-frequency band]",
    0.2, 0, 1, 0.01)) : it.remap(0, 1, 1, 8);

    t60m = freq_group(vslider("[3] Mid RT60 [style:knob] [tooltip:
        T60 = time (in seconds) to decay 60dB in middle band]", .2, 0, 1, 0.01)) : it.remap(0, 1, 1, 8);

    // f2 = freq_group(vslider("[4] HF Damping [unit:Hz] [style:knob] [scale:log]
    // [tooltip: Frequency (Hz) at which the high-frequency T60 is half the middle-band's T60]",
    // 6000, 1500, 0.49*fsmax, 1));

    f2 = freq_group(vslider("[4] HF Damping [style:knob]
    [tooltip: Frequency (Hz) at which the high-frequency T60 is half the middle-band's T60]",
    0.5, 0, 1, .01)) : hzScale(1500, 0.49*fsmax);

    out_eq = pareq_stereo(eq1f,eq1l,eq1q) : pareq_stereo(eq2f,eq2l,eq2q);
    // Zolzer style peaking eq (not used in zita-rev1) (filters.lib):
    // pareq_stereo(eqf,eql,Q) = peak_eq(eql,eqf,eqf/Q), peak_eq(eql,eqf,eqf/Q);
    // Regalia-Mitra peaking eq with "Q" hard-wired near sqrt(g)/2 (filters.lib):
    pareq_stereo(eqf,eql,Q) = peak_eq_rm, peak_eq_rm
    with {
        tpbt = wcT/sqrt(max(0,g)); // tan(PI*B/SR), B bw in Hz (Q^2 ~ g/4)
        wcT = 2*ma.PI*eqf/ma.SR;  // peak frequency in rad/sample
        g = ba.db2linear(eql); // peak gain
        peak_eq_rm = fi.peak_eq_rm(eql,eqf,tpbt);
    };

    eq1_group(x) = fdn_group(hgroup("[3] RM Peaking Equalizer 1", x));

    // eq1f = eq1_group(vslider("[1] Eq1 Freq [unit:Hz] [style:knob] [scale:log] [tooltip:
    //     Center-frequency of second-order Regalia-Mitra peaking equalizer section 1]",
    // 315, 40, 2500, 1));
    eq1f = eq1_group(vslider("[1] Eq1 Freq [style:knob] [tooltip:
        Center-frequency of second-order Regalia-Mitra peaking equalizer section 1]",
    .2, 0, 1, .01)) : hzScale(40, 2500);

    // eq1l = eq1_group(vslider("[2] Eq1 Level [unit:dB] [style:knob] [tooltip: Peak level
    //     in dB of second-order Regalia-Mitra peaking equalizer section 1]", 0, -15, 15, 0.1));
    eq1l = eq1_group(vslider("[2] Eq1 Level [style:knob] [tooltip: Peak level
        in dB of second-order Regalia-Mitra peaking equalizer section 1]", 0.5, 0, 1, 0.01)) : it.remap(0, 1, -15, 15);

    eq1q = eq1_group(vslider("[3] Eq1 Q [style:knob] [tooltip: Q = centerFrequency/bandwidth
        of second-order peaking equalizer section 1]", 3, 0.1, 10, 0.1));

    eq2_group(x) = fdn_group(hgroup("[4] RM Peaking Equalizer 2", x));

    // eq2f = eq2_group(vslider("[1] Eq2 Freq [unit:Hz] [style:knob] [scale:log] [tooltip:
    //     Center-frequency of second-order Regalia-Mitra peaking equalizer section 2]",
    // 1500, 160, 10000, 1));
    eq2f = eq2_group(vslider("[1] Eq2 Freq [style:knob] [tooltip:
        Center-frequency of second-order Regalia-Mitra peaking equalizer section 2]",
    0.1, 0, 1, .01)) : hzScale(160, 10000);

    // eq2l = eq2_group(vslider("[2] Eq2 Level [unit:dB] [style:knob] [tooltip: Peak level
    //     in dB of second-order Regalia-Mitra peaking equalizer section 2]", 0, -15, 15, 0.1));
    eq2l = eq2_group(vslider("[2] Eq2 Level [style:knob] [tooltip: Peak level
        in dB of second-order Regalia-Mitra peaking equalizer section 2]", 0.5, 0, 1, 0.01)) : it.remap(0, 1, -15, 15);

    eq2q = eq2_group(vslider("[3] Eq2 Q [style:knob] [tooltip: Q = centerFrequency/bandwidth
        of second-order peaking equalizer section 2]", 3, 0.1, 10, 0.1));

    out_group(x) = fdn_group(hgroup("[5] Output", x));

    drywet = out_group(vslider("[1] Mix [style:knob] [tooltip: Ratio of dry and wet signal. 1 = fully wet, 0 = fully dry]",
    1, 0, 1, 0.01));
};

// process = zita_rev1;
replace = !,_;
process = [
    "In Delay": replace,
    "LF X": replace,
    "Low RT60": replace,
    "Mid RT60": replace,
    "HF Damping": replace,
    "Eq1 Freq": replace,
    "Eq1 Level": replace,
    "Eq2 Freq": replace,
    "Eq2 Level": replace,
    "Mix": replace
    -> zita_rev1
    ];


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
