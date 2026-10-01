import("stdfaust.lib");
clampremap(from1, from2, to1, to2) = aa.clip(from1, from2) : it.remap(from1, from2, to1, to2);
// bpm = hgroup("[3] Master", vslider("BPM [style:knob]", 120., 60., 240, .01));
bpm = vslider("BPM [style:knob]", 120., 60., 240, .01);
// Convert a bar to Hz. Note that b==0.25 means a quarter note.
// used in lfo.lib, fx_flanger.lib, fx_chorus.lib
bar2hz(b) = rate
with {
    rate = bpm / (4*60*b); // note: potential division by zero. Use `select2` elsewhere to prevent this.
};

// David Braun adapted this from Julius's dm.flanger_demo in the official faustlibraries.
declare fx_flanger author "Julius O. Smith III";
declare fx_flanger licence "MIT";

fx_flanger(use_bpm, _rate, _dflange, _odflange, _depth, _fb, _mix) = ef.dryWetMixer(wetAmount, flanger)
with {
    dmax = 2048;
    rate = _rate : aa.clip(0, 1);
    dflange = _dflange : clampremap(0, 1, 0, 0.001 * 20 * ma.SR);
    odflange = _odflange : clampremap(0, 1, 0, 0.001 * 20 * ma.SR);
    depth = _depth : aa.clip(0, 1);
    fb = _fb : clampremap(0, 1, -.999, .999);
    wetAmount = _mix : aa.clip(0, 1);

    lfol = os.oscrs;
    lfor = os.oscrc;

    rate_from_hz = rate : pow(_,4) : _ * 20;

    // see lfo.lib for similar code for getting a frequency
    rate_from_beats = floor(.5+rate*42) : scale
    with {
        scale(i) = ba.if(i==0, 0, pow(2, 5-floor(i/3))*ba.if((i%3)==0,1,ba.if((i%3)==1,2/3, 1/3)) : bar2hz);
    };

    final_rate = ba.if(use_bpm, rate_from_beats, rate_from_hz);

    curdel1 = odflange+dflange*(1 + lfol(final_rate))/2 : min(dmax);
    curdel2 = odflange+dflange*(1 + lfor(final_rate))/2 : min(dmax);
    invert = 0;

    flanger = pf.flanger_stereo(dmax, curdel1, curdel2, depth, fb, invert);
};

fx_flanger_ui = hgroup("Flanger", fx_flanger(use_bpm, freq, dflange, odflange, depth, fb, mix))
with {
    // use_bpm = checkbox("[0] Use BPM");
    use_bpm = 0;
    freq = hslider("[1] Freq [style:knob]", 0.5, 0, 1, 0.01);
    dflange = hslider("[2] Flange Delay [style:knob]", .5, 0, 1, 0.001);
    odflange = hslider("[3] Delay Offset [style:knob]", .05, 0, 1, 0.001);
    depth = hslider("[4] Depth [style:knob]",  .1, 0., 1., .01);
    fb = hslider("[5] Feedback [style:knob]",  0.5, 0., 1., .001);
    mix = vslider("[6] Mix [style:knob]",  1, 0., 1., .01);
};

// process = fx_flanger_ui;
process = fx_flanger(0);
