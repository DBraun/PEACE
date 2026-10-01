N_FILTERS = 18;

import("stdfaust.lib");
clampremap(from1, from2, to1, to2) = aa.clip(from1, from2) : it.remap(from1, from2, to1, to2);

// todo:
// better filter modes
// fix Q and bandwidth code
// see the appendix of the Serum manual

// These numbers are rough. The goal is roughly that freq_scale(0)==8 and freq_scale(1)==13290.
freq_scale(x) = 7.39353678992449*x + 2.1012962586591915 : exp;

fx_filter(mode, _cutoff, _res, _drive, _other, _mix) = ef.dryWetMixer(wetAmount, myfilter)
with {
    cutoff = _cutoff : aa.clip(0, 1) : freq_scale;
    res = _res       : aa.clip(0, 1);
    drive = _drive   : clampremap(0, 1, 0, 22) : ba.db2linear;
    other = _other   : aa.clip(0, 1);
    wetAmount = _mix : aa.clip(0, 1);

    filterMode(0) = fi.lowpass(1, cutoff);
    filterMode(1) = fi.lowpass(2, cutoff);
    filterMode(2) = fi.lowpass(3, cutoff);
    filterMode(3) = fi.lowpass(4, cutoff);

    filterMode(4) = fi.highpass(1, cutoff);
    filterMode(5) = fi.highpass(2, cutoff);
    filterMode(6) = fi.highpass(3, cutoff);
    filterMode(7) = fi.highpass(4, cutoff);

    filterMode(8) = fi.svf.bp(cutoff,Q);
    filterMode(9) = fi.svf.notch(cutoff,Q);

    filterMode(10) = fi.svf_notch_morph(cutoff, Q, other);

    filterMode(11) = fi.resonlp(cutoff, Q, gain);
    filterMode(12) = fi.resonhp(cutoff, Q, gain);
    filterMode(13) = fi.resonbp(cutoff, Q, gain);

    filterMode(14) = fi.lowshelf(1, gainDB, cutoff);
    filterMode(15) = fi.lowshelf(3, gainDB, cutoff);

    filterMode(16) = fi.highshelf(1, gainDB, cutoff);
    filterMode(17) = fi.highshelf(3, gainDB, cutoff);

    Q = res <: pow(_,3) : it.interpolate_linear(_, .1, 18); // todo:
    gainDB = it.interpolate_linear(other, -12, 12);
    gain = gainDB : ba.db2linear; // todo:

    // myfilter = sp.stereoize(_*drive : filterMode(mode));
    myfilter = sp.stereoize(_*drive <: par(i, N_FILTERS, filterMode(i)) : ba.selectn(N_FILTERS, mode));
};

fx_filter_ui = hgroup("FX Filter", fx_filter(mode, cutoff, res, drive, other, mix))
with {
    mode = nentry("Mode", 0, 0, N_FILTERS-1, 1);
    cutoff = vslider("[0] Cutoff [style:knob]", .5, 0, 1, .01);
    res    = vslider("[1] Res [style:knob]",    .1, 0, 1, .01);
    drive  = vslider("[2] Drive [style:knob]",   0, 0, 1, .01);
    other  = vslider("[3] Other [style:knob]",   0, 0, 1, .01);
    mix    = vslider("[4] Mix [style:knob]",     1, 0, 1, .01);
};

process = fx_filter;
// process = fx_filter_ui;
// process = no.noise <: fx_filter_ui;
