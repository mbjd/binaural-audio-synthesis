#!/usr/bin/env python3

# Python rewrite of apply_hrtf.m

# Input etc.
# {{{
import numpy as np
import scipy as sp
import scipy.io
import matplotlib.pyplot as pl

# This should be a matlab 6 compatible file (save -6 <file> <vars>
# in octave) as created by upsample_irs.m
def load_irs_and_delaydiffs(filename):
	m = sp.io.loadmat(filename)['irs_and_delaydiffs']

	class irs_and_delaydiffs:
		upsampling = m[0][0]['upsampling'][0][0]

		diffs_left = m[0][0]['diffs_left']
		diffs_right = m[0][0]['diffs_right']

		irs_left = m[0][0]['irs_left']
		irs_right = m[0][0]['irs_right']

	return irs_and_delaydiffs

irs_and_delaydiffs = load_irs_and_delaydiffs('irs_and_delaydiffs_compensated_6.mat')

# }}}

# HRTF interpolation
# {{{

# Returns the delay compensated interpolated version of a signal equivalent to:
# (1-alpha) before + alpha after, with before and after referring to indices
# in the irs_and_delaydiffs database
# irs_and_delaydiffs: class as returned by load_irs_and_delaydiffs
# before, after: integers
# alpha: float between 0 and 1
# TODO maybe use one array with an extra dimension for left and right channels?
def delay_compensated_interpolation_efficient(irs_and_delaydiffs, before: int, after: int, alpha: float) -> np.ndarray:
	upsampling = irs_and_delaydiffs.upsampling

	# get the impulse responses of the 'before' sampling point
	l_before = irs_and_delaydiffs.irs_l[before,:]
	r_before = irs_and_delaydiffs.irs_r[before,:]

	# get the delay differences for both channels, convert them back to the upsampled domain
	delay_l = upsampling * irs_and_delaydiffs.diffs_l[before, after]
	delay_r = upsampling * irs_and_delaydiffs.diffs_r[before, after]

	# get the ir's of the 'after' sampling point, and remove the delay
	l_after_nodelay = delay_signal_float(irs_and_delaydiffs.irs_l[after,:], -delay_l)
	r_after_nodelay = delay_signal_float(irs_and_delaydiffs.irs_r[after,:], -delay_r)

	# interpolate the delay-free impulse responses
	l_interpolated_nodelay = (1-alpha) * l_before + alpha * l_after_nodelay
	r_interpolated_nodelay = (1-alpha) * r_before + alpha * r_after_nodelay

	# interpolate delays and add them back to the signals
	delay_l_interpolated = alpha * delay_l
	delay_r_interpolated = alpha * delay_r

	l_interpolated = delay_signal_float(l_interpolated_nodelay, delay_l_interpolated, upsampling);
	r_interpolated = delay_signal_float(r_interpolated_nodelay, delay_r_interpolated, upsampling);

	return np.concatenate([l_interpolated, r_interpolated]).reshape([2, l_interpolated.size])


# Delay a signal by a non-integer amount of samples by interpolating
# linearly between the two signals delayed by the adjacent integers
# downsample: int specifying how much to downsample the resulting signal
# terminology: before and after are the sample points adjacent to the desired
#              interpolation of hrtf's
#              left and right refer to the left and right ears/impulse responses/channels
def delay_signal_float(in_sig: np.ndarray, samples: float, downsample = 1) -> np.ndarray:
	# two adjacent integers & linear interpolation parameter
	before = int(np.floor(samples))
	after  = int(np.ceil(samples))
	a = samples - before

	# two adjacent shifts
	# TODO maybe set the rollover part to zero, check if it sounds ok without it
	# https://github.com/nils-werner/dspy/blob/master/dspy/Operator.py
	delayed_before = np.roll(in_sig, before)
	delayed_after  = np.roll(in_sig, after)

	# 'downsample' the signal before adding together (maybe this will make it slightly faster?)
	if downsample > 1:
		s = delayed_before.size;
		delayed_before = delayed_before[np.arange(0, s, downsample)]
		delayed_after = delayed_after[np.arange(0, s, downsample)]

	return (1-a) * delayed_before + a * delayed_after

# }}}
