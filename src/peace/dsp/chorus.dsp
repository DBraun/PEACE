import("stdfaust.lib");

// mathematical hard-syncing phasor (see `phasor_imp` in `faustlibraries/oscillators.lib`)
m_hsp_phasor(freq, reset, phase) = (select2(hard_reset, +(freq/ma.SR), phase) : ma.decimal) ~ _
with {
    hard_reset = (1-1')|reset; // To correctly start at `phase` at the first sample
};

// bpm = hgroup("[3] Master", vslider("BPM [style:knob]", 120., 60., 240, .01));
bpm = vslider("BPM [style:knob]", 120., 60., 240, .01);

// Convert a bar to Hz. Note that b==0.25 means a quarter note.
// used in lfo.lib, fx_flanger.lib, fx_chorus.lib
bar2hz(b) = rate
with {
    rate = bpm / (4*60*b); // note: potential division by zero. Use `select2` elsewhere to prevent this.
};

bar2samp(b) = ma.SR * (4*60*b / bpm);

DELAY_QUALITY = 3; // [1-3]

fx_chorus(use_bpm, _rate, _delay1, _delay2, _depth, _feedback, _lpf, _wetAmount) = ef.dryWetMixer(wetAmount, FX)
with {

    DELAY_MS = 20;
    DEPTH_MS = 26;

    rate = _rate : aa.clip(0, 1) ;
    delay1 = _delay1 : aa.clip(0, 1) : pow(_,2) : _ * ma.SR * DELAY_MS / 1000;
    delay2 = _delay2 : aa.clip(0, 1) : pow(_,2) : _ * ma.SR * DELAY_MS / 1000;
    depth = _depth : aa.clip(0, 1) : pow(_,2) : _ * ma.SR * DEPTH_MS / 1000;

    feedback = _feedback : aa.clip(0, 1) : _*.95;

    lpf = _lpf : aa.clip(0, 1) : pow(_,4) : it.remap(0, 1, 50, 20000);

    wetAmount = _wetAmount : aa.clip(0, 1);

    rate_from_hz = rate : pow(_,4) : _ * 10;

    // see lfo.lib for similar code for getting a frequency
    rate_from_beats = floor(.5+rate*42) : scale
    with {
        scale(i) = ba.if(i==0, 0, pow(2, 5-floor(i/3))*ba.if((i%3)==0,1,ba.if((i%3)==1,2/3, 1/3)) : bar2hz);
    };

    final_rate = ba.if(use_bpm, rate_from_beats, rate_from_hz);

    m_lfo(freq, phase) = m_hsp_phasor(freq, 0, phase) : sin(_*2*ma.PI);

    lfo1 = m_lfo(final_rate,  0) : it.remap(-1, 1, 0, depth) : _+delay1;
    lfo2 = m_lfo(final_rate, .5) : it.remap(-1, 1, 0, depth) : _+delay2;

    filter = fi.lowpass(1, lpf); // todo: other filter?

    mydelay = de.fdelayltv(DELAY_QUALITY, (DELAY_MS+DEPTH_MS)*ma.SR/1000);

    FX = (si.bus(4) :> mydelay(lfo1), mydelay(lfo2) :> filter, filter) ~ (_*feedback,_*feedback);
};

fx_chorus_ui = hgroup("Chorus", fx_chorus(use_bpm, rate, delay1, delay2, depth, feedback, lpf, wetAmount))
with {
    use_bpm   = checkbox("[0] Use BPM");
    rate      = hslider("[1] Rate [style:knob]", 0.35, 0, 1, .01);
    delay1    = hslider("[2] Delay 1 [style:knob]", 0.5, 0, 1., .01);
    delay2    = hslider("[3] Delay 2 [style:knob]", 0, 0, 1, .01);
    depth     = hslider("[4] Depth [style:knob]", 1, 0., 1, .01);
    feedback  = hslider("[5] Feedback [style:knob]", 0.15, 0., 1., .01);
    lpf       = hslider("[6] LPF [style:knob]", 0.5, 0., 1., .01);
    wetAmount = vslider("[7] Mix [style:knob]", 1, 0., 1., .01);
};

// process = fx_chorus_ui;
process = fx_chorus(use_bpm) with {
    use_bpm = 1;
};
