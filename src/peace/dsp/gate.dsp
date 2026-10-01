import("stdfaust.lib");

gate_stereo(_thresh, _att, _hold, _rel) = ef.gate_stereo(thresh,att,hold,rel)
with {
    thresh = _thresh : it.remap(0, 1, -120, -12);
    att = _att : pow(_, 4) : it.remap(0, 1, 0, 1) : max(1.0/float(ma.SR));
    hold = _hold : pow(_, 4) : it.remap(0, 1, 0, 1) : max(1.0/float(ma.SR));
    rel = _rel : pow(_, 4) : it.remap(0, 1, 0, 1) : max(1.0/float(ma.SR));
};

gate_stereo_ui = gate_stereo(
    hslider("thresh", 0.5, 0, 1, .01),
    hslider("att", 0.5, 0, 1, .01),
    hslider("hold", 0.5, 0, 1, .01),
    hslider("rel", 0.5, 0, 1, .01)
);

process = gate_stereo;
// process = gate_stereo_ui;
// process = _, _;
