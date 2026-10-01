import("stdfaust.lib");
//import("utils.lib");
clampremap(from1, from2, to1, to2) = aa.clip(from1, from2) : it.remap(from1, from2, to1, to2);

// These numbers are rough. The goal is roughly that freq_scale(0)==8 and freq_scale(1)==13290.
freq_scale(x) = 7.39353678992449*x + 2.1012962586591915 : exp;

softclip(drive) = _*drive : aa.softclipQuadratic2 : _*2/3;

hardclip(drive) = _*drive : aa.hardclip2;

cubic0(x_f) = ba.if( (x_f < 1.0) & (x_f > -1.0),
                           x_f - x_f ^ 3.0 / 3.0,
                           (2.0 / 3.0) * ma.signum(x_f));

// cubic(drive) = _*drive : aa.cubic1; // has artifacts, don't use.
cubic(drive) = _*drive : cubic0;

wavefold(drive) = ef.wavefold(drive);

tanh_distortion(drive) = _*drive : aa.tanh1;

tubes = library("tubes.lib");

T1_12AX7(drive) = _*drive : tubes.T1_12AX7;
T2_12AX7(drive) = _*drive : tubes.T2_12AX7;
T3_12AX7(drive) = _*drive : tubes.T3_12AX7;
T1_12AT7(drive) = _*drive : tubes.T1_12AT7;
T2_12AT7(drive) = _*drive : tubes.T2_12AT7;
T3_12AT7(drive) = _*drive : tubes.T3_12AT7;

T1_12AU7(drive) = _*drive : tubes.T1_12AU7;
T2_12AU7(drive) = _*drive : tubes.T2_12AU7;
T3_12AU7(drive) = _*drive : tubes.T3_12AU7;
T1_6V6(drive) = _*drive : tubes.T1_6V6;
T2_6V6(drive) = _*drive : tubes.T2_6V6;
T3_6V6(drive) = _*drive : tubes.T3_6V6;

T1_6DJ8(drive) = _*drive : tubes.T1_6DJ8;
T2_6DJ8(drive) = _*drive : tubes.T2_6DJ8;
T3_6DJ8(drive) = _*drive : tubes.T3_6DJ8;
T1_6C16(drive) = _*drive : tubes.T1_6C16;
T2_6C16(drive) = _*drive : tubes.T2_6C16;
T3_6C16(drive) = _*drive : tubes.T3_6C16;

// downsample(drive) = ba.sAndH(hold)
// with {
//     // If amt is 0, then freq should be ma.SR/2.
//     // If amt is 1, then freq should be 491.
//     // 491 is derived from listening to the Serum synthesizer.
//     freq = drive : it.remap(0, 1, ba.hz2midikey(ma.SR/2), ba.hz2midikey(491)) : ba.midikey2hz;
//     hold = ba.time%int(ma.SR/freq) == 0;
// };

downsample(drive) = ba.downSampleCV(drive);

// todo: add more nonlinearities.

// Tube
// Diode 1
// Diode 2
// Sine Fold
// Zero-Square
// Asym
// Rectify
// Sine Shaper
// Stomp Box
// Tape Saturator

N_MODES = 6;


fx_distortion(_filtermode, nl_mode, _freq, _q, _style, _drive, _mix) = ef.dryWetMixer(wetAmount, sp.stereoize(distAndFilter))
with {

    freq = _freq     : aa.clip(0, 1) : freq_scale;
    q = _q           : clampremap(0, 1, 0.01, 1); // todo:
    style = _style   : aa.clip(0, 1) * 2; // [0: lowpass; 1: bandpass; 2: highpass]
    drive = _drive   : clampremap(0, 1, -6, 30) : ba.db2linear; // todo scale:
    wetAmount = _mix : aa.clip(0, 1);

    distortionFn = _ <:
        softclip(drive),
        hardclip(drive),
        cubic(drive),
        wavefold(_drive),
        tanh_distortion(drive),
        downsample(_drive)
        // ve.klonCentaur(drive, 0.5, 1.0)
        // T1_12AX7(drive), // bad
        // T2_12AX7(drive),
        // T3_12AX7(drive),
        // T1_12AT7(drive),
        // T2_12AT7(drive),
        // T3_12AT7(drive),

        // T1_12AU7(drive),
        // T2_12AU7(drive),
        // T3_12AU7(drive),
        // T1_6V6(drive),
        // T2_6V6(drive),
        // T3_6V6(drive),

        // T1_6DJ8(drive),
        // T2_6DJ8(drive),
        // T3_6DJ8(drive),
        // T1_6C16(drive),
        // T2_6C16(drive),
        // T3_6C16(drive)
        : ba.selectn(N_MODES, nl_mode);

    SVF = fi.svf_notch_morph(freq, q, style);

    distAndFilter = _ <:
        (distortionFn), // off
        (SVF : distortionFn), // filter pre
        (distortionFn : SVF) // filter post
        : select3(_filtermode);
};

fx_distortion_ui = hgroup("Distortion", fx_distortion(filterMode, nlMode, freq, q, style, drive, mix))
with {
    filterMode = nentry("[0] Filter Mode [style:menu{'Off':0;'Pre':1,'Post':2}]", 0, 0, 2, 1);
    nlMode = nentry("[1] Distortion Mode", 0, 0, N_MODES-1, 1);
    freq  = vslider("[2] Freq [style:knob]",  1, 0., 1., .01);
    q     = vslider("[3] Q [style:knob]",     1, 0., 1., .01);
    style = vslider("[4] Style [style:knob]", 0, 0., 1., .01);
    drive = vslider("[5] Drive [style:knob]", it.remap(-6,30,0,1,0), 0, 1, .01);
    mix   = vslider("[6] Mix [style:knob]",   1, 0., 1., .01);
};

process = fx_distortion;
// process = fx_distortion_ui;
// process = fx_distortion(filterMode, nlMode) with {
//     filterMode = 1; // ['off', 'filter-pre', 'filter-post']
//     nlMode = 0; // ['softclip', 'hardclip']
// };
