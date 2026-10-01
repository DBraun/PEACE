import("stdfaust.lib");

// re.vital_rev expects 12 parameter channels + 2 audio channels (L/R)
// Parameters: prelow, prehigh, lowcutoff, highcutoff, lowGain, highGain,
//            chorus_amt, chorus_freq, predelay, time, size, mix
process = re.vital_rev;
