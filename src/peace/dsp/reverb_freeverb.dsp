import("stdfaust.lib");

hzScale(lowHz, highHz) = it.remap(0, 1, ba.hz2midikey(lowHz), ba.hz2midikey(highHz)) : ba.midikey2hz;

//======================================Reverbs===========================================
//========================================================================================

//----------------------------`(dm.)freeverb_demo`-------------------------
// Freeverb demo application.
//
// #### Usage
//
// ```
// _,_ : freeverb_demo : _,_
// ```
//
// #### Test
// ```
// dm = library("demos.lib");
// os = library("oscillators.lib");
// freeverb_demo_test = os.osc(440), os.osc(442) : dm.freeverb_demo;
// ```
//------------------------------------------------------------
declare freeverb_demo author " Romain Michon";
declare freeverb_demo licence "LGPL";

freeverb_demo = dryWetMixerConstantPower(wet, g : re.stereo_freeverb(combfeed, allpassfeed, damping, spatSpread))
with{
    g = *(0.2), *(0.2); // DBraun picked this
    scaleroom   = 0.28;
    offsetroom  = 0.7;
    allpassfeed = 0.5;
    scaledamp   = 0.4;
    origSR = 44100;

    parameters(x) = hgroup("Freeverb",x);
    knobGroup(x) = parameters(vgroup("[0]",x));
    damping = knobGroup(vslider("[0] Damp [style: knob] [tooltip: Somehow control the
        density of the reverb.]",0.5, 0, 1, 0.025)*scaledamp*origSR/ma.SR);
    combfeed = knobGroup(vslider("[1] RoomSize [style: knob] [tooltip: The room size
        between 0 and 1 with 1 for the largest room.]", 0.5, 0, 1, 0.025)*scaleroom*
        origSR/ma.SR + offsetroom);
    spatSpread = knobGroup(vslider("[2] Stereo Spread [style: knob] [tooltip: Spatial
        spread between 0 and 1 with 1 for maximum spread.]",0.5,0,1,0.01)*46*ma.SR/origSR
        : int);
    wet = parameters(vslider("[1] Mix [tooltip: The amount of reverb applied to the signal
        between 0 and 1 with 1 for the maximum amount of reverb.]", 0.3333, 0, 1, 0.01));
};

// process = freeverb_demo;
replace = !, _;
process = ["Damp": replace, "RoomSize": replace, "Stereo Spread": replace, "Mix": replace -> freeverb_demo];

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
