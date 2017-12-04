#!/usr/bin/env python3

# Python rewrite of apply_hrtf.m

# Input etc.
# {{{
import time

import numpy as np
import scipy as sp

import scipy.io
import scipy.io.wavfile as wavfile
import scipy.signal

import matplotlib.pyplot as pl

# This should be a matlab 6 compatible file (save -6 <file> <vars>
# in octave) as created by upsample_irs.m
def load_irs_and_delaydiffs(filename = 'irs_and_delaydiffs_compensated_6.mat'):
	m = sp.io.loadmat(filename)['irs_and_delaydiffs']

	class irs_and_delaydiffs:
		upsampling = m[0][0]['upsampling'][0][0]

		diffs_left = m[0][0]['diffs_left']
		diffs_right = m[0][0]['diffs_right']

		irs_left = m[0][0]['irs_left']
		irs_right = m[0][0]['irs_right']

	return irs_and_delaydiffs

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
# this is the same as delay_compensated_interpolation_efficient in the octave version
def delay_compensated_interpolation(irs_and_delaydiffs, before: int, after: int, alpha: float):
	upsampling = irs_and_delaydiffs.upsampling

	# get the impulse responses of the 'before' sampling point
	l_before = irs_and_delaydiffs.irs_left[before,:]
	r_before = irs_and_delaydiffs.irs_right[before,:]

	# get the delay differences for both channels, convert them back to the upsampled domain
	delay_l = upsampling * irs_and_delaydiffs.diffs_left[before, after]
	delay_r = upsampling * irs_and_delaydiffs.diffs_right[before, after]

	# get the ir's of the 'after' sampling point, and remove the delay
	l_after_nodelay = delay_signal_float(irs_and_delaydiffs.irs_left[after,:], -delay_l)
	r_after_nodelay = delay_signal_float(irs_and_delaydiffs.irs_right[after,:], -delay_r)

	# interpolate the delay-free impulse responses
	l_interpolated_nodelay = (1-alpha) * l_before + alpha * l_after_nodelay
	r_interpolated_nodelay = (1-alpha) * r_before + alpha * r_after_nodelay

	# interpolate delays and add them back to the signals
	delay_l_interpolated = alpha * delay_l
	delay_r_interpolated = alpha * delay_r

	l_interpolated = delay_signal_float(l_interpolated_nodelay, delay_l_interpolated, int(upsampling));
	r_interpolated = delay_signal_float(r_interpolated_nodelay, delay_r_interpolated, int(upsampling));

	return np.concatenate([l_interpolated, r_interpolated]).reshape([2, l_interpolated.size])

# Same but with different interface
def delay_compensated_interpolation_easy(irs_and_delaydiffs, continuous_index: float):
	before = int(np.floor(continuous_index))
	after = int(np.ceil(continuous_index))
	alpha = continuous_index - before

	# TODO remove this once we can do 2d interpolation
	# also this only works when the signal does circles counterclockwise
	if (after == 97):
		after = 73

	return delay_compensated_interpolation(irs_and_delaydiffs, before, after, alpha)

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

# Application of interpolated HRTF's
# {{{

def make_signal_move(in_signal, chunksize: int, index_function, irs_and_delaydiffs):
	'''
	in_signal: input signal, ndarray of shape (1, N) (TODO: (2,N) for stereo signals)
	chunksize: Number of samples for which to use the same HRTF interpolation
	           (TODO: make this depend on the derivative of index_function)
	index_function: function of time (in samples) specifying the index of the HRTF to use
	irs_and_delaydiffs: class returned by load_irs_and_delaydiffs
	'''

	assert len(in_signal.shape) == 1, 'only mono signals for now'
	ir_length = int(0.5 + irs_and_delaydiffs.irs_left.shape[1] / irs_and_delaydiffs.upsampling)

	in_length = int(0.5 + np.ceil(in_signal.size / chunksize) * chunksize)
	in_signal = np.pad(in_signal, (0, in_length - in_signal.size), mode='constant')
	assert in_signal.size == in_length, 'input has been padded with wrong nubmer of zeros'

	# output length = next bigger integer multiple of chunk size + ir length in worst case
	out_length = int(0.5 + np.ceil(in_signal.size / chunksize) * chunksize + (ir_length - 1))
	assert out_length == in_signal.size + ir_length - 1, 'wrong output length'

	out_l = np.zeros([out_length])
	out_r = np.zeros([out_length])

	# create the chunks here to avoid reallocating them every time
	# probably pretty pointless since the interpolation function
	# reallocates the upsampled IR's anyway
	out_chunk_left = np.zeros([chunksize + ir_length - 1])
	out_chunk_right = np.zeros([chunksize + ir_length - 1])
	out_indices = np.zeros([chunksize + ir_length - 1], dtype=np.uint64)

	'''
	IDEA

	to make this more efficient for a given resolution, use chunks and
	sub-chunks: each chunk has a newly computed delay compensated
	interpolated IR at the start and end, and within the chunk, the IR is
	simply interpolated linearly between the two actual interpolations for
	each sub-chunk. The chunks would have to be small enough that the delay
	difference between the two adjacent interpolated impulse response (at
	the start of one chunk and the next one) is much smaller than one
	sample.

	For example, we could use chunks of 400 and sub-chunks of 10 samples
	(400 samples = 9 ms, during which a sound source would have to travel
	at an angular velocity of (...) to exceed a difference in delay
	difference more than 1 sample)

	(TODO: actually calculate this and maybe implement a dynamic algorithm
	based on the difference quotient of index_function)

	'''

	for i in range(0, in_length, chunksize):
		in_chunk = in_signal[i:i+chunksize];
		ir_interpolated = delay_compensated_interpolation_easy(irs_and_delaydiffs, index_function(i))

		out_indices[:] = np.arange(i, i + chunksize + ir_length - 1)

		out_chunk_left[:]  = sp.signal.convolve(in_chunk, ir_interpolated[0,:])
		out_chunk_right[:] = sp.signal.convolve(in_chunk, ir_interpolated[1,:])

		out_l[out_indices] += out_chunk_left;
		out_r[out_indices] += out_chunk_right;

		if ((i // chunksize) % 64 == 0):
			print('{:.1f}%           '.format(100 * i / in_length), end='\r')
	print('100.0%')

	out_sig = np.concatenate([out_l, out_r]).reshape([2, out_l.size]).T
	return out_sig / np.max([out_sig.max(), -(out_sig.min())])
# }}}

def main():
	# needs to be mono
	start = time.time()
	fs, y = wavfile.read('netzwerk.wav')
	assert(len(y.shape) == 1)
	y = y.astype(np.float32) / y.max()

	irs_and_delaydiffs = load_irs_and_delaydiffs('irs_and_delaydiffs_compensated_6.mat')

	T=1 # Period of signal moving around head
	chunksize = 10
	out_sig = make_signal_move(y, chunksize, lambda t: (t % (T*fs)) * (24 / (T*fs)) + 73, irs_and_delaydiffs).astype(np.float32);

	out_filename = 'python-netzwerk-{}.wav'.format(chunksize);
	wavfile.write(out_filename, fs, out_sig.astype(np.float32))
	print("wrote to '{}' - took {:.2f} secs".format(out_filename, time.time() - start))

if __name__ == '__main__':
	main()
