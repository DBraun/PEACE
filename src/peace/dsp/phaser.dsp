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

hzScale(lowHz, highHz) = it.remap(0, 1, ba.hz2midikey(lowHz), ba.hz2midikey(highHz)) : ba.midikey2hz;

// David Braun adapted this from Julius's dm.flanger_demo in the official faustlibraries.
declare fx_phaser author "Julius O. Smith III";
declare fx_phaser licence "MIT";

fx_phaser(use_bpm, _rate, _notchDepth, _fb, _notchWidth, _minNotchFreq, _maxNotchFreq, _notchFreqRatio, _mix) = ef.dryWetMixer(wetAmount, phaser)
with {
    rate = _rate : aa.clip(0, 1);
    notchDepth = _notchDepth : aa.clip(0, 1);
    fb = _fb : clampremap(0, 1, -.999, .999);
    notchWidth = _notchWidth : aa.clip(0, 1) : hzScale(10, 5000);
    minNotchFreq = _minNotchFreq : aa.clip(0, 1) : hzScale(20, 5000);
    maxNotchFreq = _maxNotchFreq : aa.clip(0, 1) : hzScale(20, 10000);
    notchFreqRatio = _notchFreqRatio : clampremap(0, 1, 1.1, 4);
    wetAmount = _mix : aa.clip(0, 1);

    rate_from_hz = rate : pow(_,4) : _ * 10;

    // see lfo.lib for similar code for getting a frequency
    rate_from_beats = floor(.5+rate*42) : scale
    with {
        scale(i) = ba.if(i==0, 0, pow(2, 5-floor(i/3))*ba.if((i%3)==0,1,ba.if((i%3)==1,2/3, 1/3)) : bar2hz);
    };

    speed = ba.if(use_bpm, rate_from_beats, rate_from_hz);

    invert = 0;
    Notches = 4;

    phaser = pf.phaser2_stereo(Notches,notchWidth,minNotchFreq,notchFreqRatio,maxNotchFreq,rate,notchDepth,fb,invert);
    // phaser = pf.phaser2_stereo(Notches,notchWidth,minNotchFreq,notchFreqRatio,maxNotchFreq,rate,notchDepth,fb,invert);
};

fx_phaser_ui = hgroup("Flanger", fx_phaser(use_bpm, freq, notchDepth, fb, notchWidth, minNotchFreq, maxNotchFreq, notchFreqRatio, mix))
with {
    // use_bpm = checkbox("[0] Use BPM");
    use_bpm = 0;
    freq = hslider("[1] Freq [style:knob]", 0.5, 0, 1, 0.01);
    notchDepth = hslider("[2] Notch Depth [style:knob]", .0, 0, 1, 0.001);
    fb = hslider("[3] Feedback [style:knob]",  0.5, 0., 1., .001);
    notchWidth = hslider("[4] Notch Width [style:knob]", .5, 0, 1, 0.001);
    minNotchFreq = hslider("[5] Min Notch Freq [style:knob]", .05, 0, 1, 0.001);
    maxNotchFreq = hslider("[6] Max Notch Freq [style:knob]", .85, 0, 1, 0.001);
    notchFreqRatio = hslider("[7] Notch Freq Ratio [style:knob]", 0.5, 0, 1, .001);
    mix = vslider("[8] Mix [style:knob]",  1, 0., 1., .01);
};

process = fx_phaser(0);
// process = fx_phaser_ui;
// process = dm.phaser2_demo;
