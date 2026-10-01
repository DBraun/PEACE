import("stdfaust.lib");
bpf = ba.bpf;
clampremap(from1, from2, to1, to2) = aa.clip(from1, from2) : it.remap(from1, from2, to1, to2);
// bpm = hgroup("[3] Master", vslider("BPM [style:knob]", 120., 60., 240, .01));
bpm = vslider("BPM [style:knob]", 120., 60., 240, .01);
bar2samp(b) = ma.SR * (4*60*b / bpm);

// todo: bandwidth code

fx_delay = environment {

    freq_scale = aa.clip(0,1) : _*6.109331407656626 + 3.6888004699964716 : exp;

    LAGRANGE_ORDER = 3;
    MIN_BPM = 60;
    MAX_DELAY_SEC = 0.5;
    MAX_DELAY_SAMP = ma.SR * max(4 * 240 / MIN_BPM, MAX_DELAY_SEC * 1.5); // max delay in samples (4 bars)

    delayUnit(use_bpm, delayAmt, scale) = myDelay(LAGRANGE_ORDER, MAX_DELAY_SAMP, delayDurSamp)
    with {
        bars_durs = waveform{.001953125, .00390625, .0078125, .015625, 0.03125, 0.0625, .125, .25, .5, 1, 2, 4}; // bars

        // measured in samples
        dur_samp_from_bpm = floor(.5+delayAmt*11) : rdtable(bars_durs) : bar2samp;
        dur_samp_from_ms = delayAmt : pow(_, 4) : it.remap(0, 1, 0, MAX_DELAY_SEC) : _*ma.SR;

        delayDurSamp = ba.if(use_bpm, dur_samp_from_bpm, dur_samp_from_ms)*scale;
    };

    myDelay(N, MAXDELAY, delayAmt) = de.fdelaylti(N, MAXDELAY, (delayAmt:aa.clip(MINDELAY,MAXDELAY)))
    with {
        MINDELAY = ceil((N-1)/2);
    };

    delayFactor(x) = x : bpf.start(0,.5) : bpf.point(1/11,.5) : bpf.point(2/11,2/3) : bpf.point(3/11,2/3) : bpf.point(4/11,.75) : bpf.point(5/11,.75) : bpf.point(6/11,1) : bpf.point(7/11,1) : bpf.point(8/11,4/3) : bpf.point(9/11,4/3) : bpf.point(10/11,1.5) : bpf.end(1,1.5);

    _fx_delay(fdback_method, use_bpm, link, _feedback, _delayL, _delayLScale, _delayR, _delayRScale, _freq, _q, _mix) = ef.dryWetMixer(wetAmount, FX)
    with {
        fdback_amt = _feedback : aa.clip(0, 1);
        delayL = _delayL : aa.clip(0, 1);
        delayR = _delayR : aa.clip(0, 1) : ba.if(link, delayL, _);
        delayLScale = _delayLScale : delayFactor;
        delayRScale = _delayRScale : delayFactor : ba.if(link, delayLScale, _);
        cutoff = _freq : freq_scale;
        bw = _q : clampremap(0, 1, 0, 130);
        wetAmount = _mix : aa.clip(0, 1);

        FX = (si.bus(4) :> (delayUnit(use_bpm, delayL, delayLScale), delayUnit(use_bpm, delayR, delayRScale)) : filters) ~ (fdbacks : fdback_method)
        with {
            filters = filter, filter
            with {
                // todo: bandwidth?
                low_freq = cutoff * (ba.semi2ratio(0-bw)) : max(20);
                high_freq = cutoff * (ba.semi2ratio(bw)) : min(20000);
                filter = fi.lowpass(1, high_freq) : fi.highpass(1, low_freq);
            };

            fdbacks = par(i, 2, _*fdback_amt);
        };
    };

    tap(use_bpm, link, _feedback, _delayL, _delayLScale, _delayR, _delayRScale, _freq, _q, _mix) = ef.dryWetMixer(wetAmount, FX)
    with {
        fdback_amt = _feedback : aa.clip(0, 1);
        delayL = _delayL : aa.clip(0, 1);
        delayR = _delayR : aa.clip(0, 1) : ba.if(link, delayL, _);
        delayLScale = _delayLScale : delayFactor;
        delayRScale = _delayRScale : delayFactor : ba.if(link, delayLScale, _);
        cutoff = _freq : freq_scale;
        bw = _q : clampremap(0, 1, 0, 130);
        wetAmount = _mix : aa.clip(0, 1);

        // todo: this isn't correct. the first tap should happen once. the second tap should feedback to itself.
        FX = si.bus(2) :> ((si.bus(2) :> filter : delayUnit(use_bpm, delayL, delayLScale) <: (delayUnit(use_bpm, delayR, delayRScale),_)) ~ (fdbacks)) <: ro.cross1n(1)
        with {
            // todo: bandwidth?
            low_freq = cutoff * (ba.semi2ratio(0-bw)) : max(20);
            high_freq = cutoff * (ba.semi2ratio(bw)) : min(20000);
            filter = fi.lowpass(1, high_freq) : fi.highpass(1, low_freq);

            fdbacks = _*fdback_amt;
        };
    };

    normal = _fx_delay(si.bus(2));
    ping_pong = _fx_delay(ro.crossn1(1));

};

fx_delay_ui(func) = hgroup("Delay", func(use_bpm, link, fdback_amt, delayL, delayLScale, delayR, delayRScale, freq, bw, wetAmount))
with {

    // SMOO = si.smoo;
    SMOO = _;
    // use_bpm     = checkbox("h:[1] Delay /[0] Use BPM");
    use_bpm = 0;
    link        = checkbox("h:[1] Delay /[1] Link");
    fdback_amt  = hslider("[0] Feedback [style:knob]", 0.4, 0, 1, .01);
    delayL      = hslider("h:[1] Delay /[2] Delay L [style:knob]", .05, 0., 1., .01) : SMOO;
    delayLScale = hslider("h:[1] Delay /[3] Delay L Scale [style:knob]", 6.5/11, 0., 1., .01) : SMOO;
    delayR      = hslider("h:[1] Delay /[4] Delay R [style:knob]", .05, 0., 1., .01) : SMOO;
    delayRScale = hslider("h:[1] Delay /[5] Delay R Scale [style:knob]", 6.5/11, 0., 1., .01) : SMOO;
    freq        = hslider("[2] Freq [style:knob]", .5, 0,1, .01) : SMOO;
    bw          = hslider("[3] Bandwidth [style:knob]", .05, 0., 1., .01) : SMOO;
    wetAmount   = vslider("[4] Mix [style:knob]", 1, 0., 1., .01) : SMOO;
};

fx_delay_ui(func) = hgroup("Delay", func(use_bpm, link, fdback_amt, delayL, delayLScale, delayR, delayRScale, freq, bw, wetAmount))
with {

    // SMOO = si.smoo;
    SMOO = _;
    // use_bpm     = checkbox("h:[1] Delay /[0] Use BPM");
    use_bpm = 0;
    link        = checkbox("h:[1] Delay /[1] Link");
    fdback_amt  = hslider("[0] Feedback [style:knob]", 0.4, 0, 1, .01);
    delayL      = hslider("h:[1] Delay /[2] Delay L [style:knob]", .05, 0., 1., .01) : SMOO;
    delayLScale = hslider("h:[1] Delay /[3] Delay L Scale [style:knob]", 6.5/11, 0., 1., .01) : SMOO;
    delayR      = hslider("h:[1] Delay /[4] Delay R [style:knob]", .05, 0., 1., .01) : SMOO;
    delayRScale = hslider("h:[1] Delay /[5] Delay R Scale [style:knob]", 6.5/11, 0., 1., .01) : SMOO;
    freq        = hslider("[2] Freq [style:knob]", .5, 0,1, .01) : SMOO;
    bw          = hslider("[3] Bandwidth [style:knob]", .05, 0., 1., .01) : SMOO;
    wetAmount   = vslider("[4] Mix [style:knob]", 1, 0., 1., .01) : SMOO;
};

fx_delay_ui_ping_pong = fx_delay_ui(fx_delay.ping_pong);
fx_delay_ui_normal = fx_delay_ui(fx_delay.normal);
fx_delay_ui_tap = fx_delay_ui(fx_delay.tap);

// choice parameter [0-2], 8 parameters, 2 bus in
process(choice) = use_bpm, link, si.bus(8), si.bus(2) <: fx_delay.ping_pong, fx_delay.normal, fx_delay.tap : ba.selectbus(2, 3, choice)
with {
    use_bpm = 0;
    link = 0;
};

// process = _*0,_ : fx_delay_ui_ping_pong;
// process = fx_delay_ui_normal;
// process = fx_delay_ui_tap;
