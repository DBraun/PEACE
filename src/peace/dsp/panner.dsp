import("stdfaust.lib");

// Constant-power stereo panner with sqrt(2) compensation
// Preserves total power for balanced stereo input (L² ≈ R²)
// pan=0: hard left, pan=0.5: center, pan=1: hard right
constantPowerPan(p, x, y) = x * gainLeft * comp, y * gainRight * comp
with {
    theta = ma.PI * p / 2.0;
    gainLeft = cos(theta);
    gainRight = sin(theta);
    comp = sqrt(2.0);
};

process = constantPowerPan;

// process = _,_;
// process = constantPowerPan(hslider("pan", 0.5, 0, 1, .01));
