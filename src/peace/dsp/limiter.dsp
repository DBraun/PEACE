import("stdfaust.lib");

// all parameters are normalized [0-1] and assumed to be within those bounds.
limiter_stereo(_gain, _ceiling, _att, _hold, _rel) =
    _*gain, _*gain
    : co.limiter_lad_stereo(LD, ceiling, attack, hold, release)
with {
    LD = 1.5 / 1000; // 1.5 ms in seconds units
    gain = _gain : it.remap(0, 1, -12, 12) : ba.db2linear;
    ceiling = _ceiling : it.remap(0, 1, -24, 0) : ba.db2linear;
    attack = _att : pow(_, 2) : it.remap(0, 1, 0, 10) : _/1000; // todo: reasonable output range?
    hold = _hold : pow(_, 2) : it.remap(0, 1, 0, 50) : _/1000; // todo: reasonable output range?
    release = _rel : pow(_, 4) : it.remap(0, 1, 0.001, 3); // todo: reasonable output range?
};

limiter_stereo_ui = hgroup("Limiter", limiter_stereo(
    vslider("[0] Gain", 0.5, 0, 1, .01),
    vslider("[1] Ceiling", 0.5, 0, 1, .01),
    vslider("[2] Attack [style:knob]", 0.5, 0, 1, .01),
    vslider("[3] Hold [style:knob]", 0.5, 0, 1, .01),
    vslider("[4] Release [style:knob]", 0.5, 0, 1, .01)
));

process = limiter_stereo;
// process = limiter_stereo_ui;
// process = os.osc(440)*1 <: limiter_stereo_ui;
// process = _, _;
