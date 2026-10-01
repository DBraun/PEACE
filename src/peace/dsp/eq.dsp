import("stdfaust.lib");
clampremap(from1, from2, to1, to2) = aa.clip(from1, from2) : it.remap(from1, from2, to1, to2);

// todo:
// fix Q and bandwidth code

fx_eq(_modeL, _freqL, _qL, _gainL, _modeH, _freqH, _qH, _gainH, _mix) = ef.dryWetMixer(wetAmount, eq)
with {

    // from roughly freq_scale(0)==22 to freq_scale(1)==20000
    freq_scale = aa.clip(0, 1) : _*6.833865314616473 + 3.0696181455818903 : exp;

    freqL = _freqL : freq_scale;
    freqH = _freqH : freq_scale;

    bandL   = _qL : aa.clip(0, 1) : freqL/it.interpolate_linear(_,10,2); // todo:
    bandH   = _qH : aa.clip(0, 1) : freqH/it.interpolate_linear(_,10,2); // todo:

    qL = _qL : aa.clip(0, 1) : it.interpolate_linear(_,.1, 8); // todo:
    qH = _qH : aa.clip(0, 1) : it.interpolate_linear(_,.1, 8); // todo:

    gainL = _gainL : clampremap(0, 1, -24, 24);
    gainH = _gainH : clampremap(0, 1, -24, 24);

    eq = sp.stereoize(filterL : filterH);

    filterL = _ <: fi.lowshelf(3,gainL,freqL), fi.peak_eq(gainL,freqL,bandL), fi.svf.hp(freqL, qL) : select3(_modeL);

    filterH = _ <: fi.highshelf(3,gainH,freqH), fi.peak_eq(gainH,freqH,bandH), fi.svf.lp(freqH, qH) : select3(_modeH);

    wetAmount = _mix : aa.clip(0, 1);
};

fx_eq_ui = hgroup("EQ", fx_eq(modeL, freqL, qL, gainL, modeH, freqH, qH, gainH, mix))
with {
    Low(x) = hgroup("[0] Low", x);
    High(x) = hgroup("[1] High", x);

    freqpar = hslider("[1] Freq [style:knob]",  .5, 0., 1., .01);
    qpar = hslider("[2] Q [style:knob]",  .5, 0., 2., .01);
    gainpar = hslider("[3] Gain [style:knob]",  .5, 0., 1., .01);

    modeL = Low(nentry("[0] Mode [style:menu{'Shelf':0;'Peak':1;'HP':2}]", 1, 0, 2, 1));
    freqL = Low(freqpar);
    qL    = Low(qpar);
    gainL = Low(gainpar);

    modeH = High(nentry("[0] Mode [style:menu{'Shelf':0;'Peak':1;'LP':2}]", 1, 0, 2, 1));
    freqH = High(freqpar);
    qH    = High(qpar);
    gainH = High(gainpar);

    mix = vslider("[2] Mix [style:knob]",  1, 0., 1., .01);
};

// process = fx_eq_ui;
process = fx_eq;
