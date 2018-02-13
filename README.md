Binaual audio synthesis
=======================

This is my (Balduin Dettling, dbalduin (at) student.ethz.ch) binaural audio
generation project I made in the digital audio P&S in late 2017. The goal
is to take an audio file, along with a function that maps time to a point
on the unit sphere, and make another audio file that sounds as if the sound
source was moving around the listener according to the given function.

For this, I need measurements of HRTFs (head related transfer functions -
the impulse responses of the human ears and head) from different directions
(1). Luckily, these measurements have already been taken several times - in
this project I used a database of HRTF measurements spaced about 15
degrees. It is to be noted, however, that ear and head shapes differ
between people, and the only way to synthesise very realistic binaural
audio is to measure your own head's impulse response from lots of different
directions.

The main challenge of the project then became interpolation, that is
creating a HRTF for a position where we don't have a measurement. The
simplest thing to do would be linearly interpolating between adjacent
HRTFs, which didn't work well because the difference in signal delay can
cause unwanted constructive or destructive interference.

Another, slightly more advanced approach, which is now used in the project,
is what we call "delay compensated interpolation" (2). We first determine
the delay difference (3) between two signals between which we want to
interpolate. Then we remove the delay in the second signal (which is "just"
a shift in time - see next paragraph), so that the signals have the same
delay and delay difference 0. We then linearly interpolate between these
two new signals, according to an interpolation parameter alpha ∈ [0,1]. To
the interpolated signal we add back an interpolated delay, which is the
linear interpolation between zero (the delay of the first signal relative
to the first signal) and the delay difference between the two original
signals (the delay of the second signal relative to the first signal).

In order to not have to use the expensive sinc interpolation formula every
time we need to shift a signal in time by a non-integer amount of samples,
we store upsampled versions of the HRTFs. The script `./upsample_irs.m`
calculates them beforehand, and they are loaded into the python script
`./apply_hrtf.py` with the function `load_irs_and_delaydiffs`.

Then we use linear interpolation on the upsampled signal to shift it in
time, and downsample simply by taking every N-th sample (assuming that the
linear interpolation doesn't introduce frequencies higher than those of the
original signal - TODO: find out if that's actually the case). Currently,
the upsampling factor N is set to 8, and other values haven't really been
tested, but in theory this value is not hardcoded anywhere and a change in
N should be as simple as generating a new `irs_and_delaydiffs` struct with
`./upsample_irs.m`.

Another small optimisation is that we cut off the impulse response after
some number of samples. If you plot the HRTFs, you see that (at least) the
latter two thirds is mostly noise, so we can safely get rid of that. The
`load_irs_and_delaydiffs` function in `./apply_hrtf.py` takes a parameter
`samples_to_keep`, which does what it says.

Using this delay compensated interpolation method, I then constructed a 2d
interpolation function, `interpolate_2d` in `./apply_hrtf.py`, which is
explained in a comment in the function itself.

Now we have a method to get a decent HRTF for any point on the sphere
(well, except if the elevation angle is below -45º, because the database
doesn't have measurements there). The rest of the challenge consists in
applying these HRTFs to a signal with a moving sound source. I experimented
with several methods, and have arrived at the abomination that is
`make_signal_move_2d` (but it's a pretty fast abomination :). It is also
explained in a comment within the function itself.

(1): If you want to do a slightly simpler project or one that uses less
resources, consider doing away with HRTFs and just try to recreate the
amplitude and delay differences between the ears from all possible angles.

(2): Implementation in `delay_compensated_interpolation_with_delaydiff` in
`./apply_hrtf.py`.

(3): The delay difference of two signals is, loosely speaking, the distance
between the positions of the main peaks in the respective HRTF. The actual
definition is its implementation (Salamon would be proud of me), which is
the function `delaydifference` in `./upsample_irs.m,` where we
cross-correlate the two impulse responses and return the position of the
peak of that signal in comparison to the middle of it (where the peak would
be if the two signals were the same).

