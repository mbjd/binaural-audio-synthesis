#!/usr/bin/env python3

# Python rewrite of apply_hrtf.m

# Input etc.
# {{{
import time
import sys

import numpy as np
import scipy as sp

import scipy.io
import scipy.io.wavfile as wavfile
import scipy.signal

import matplotlib.pyplot as pl

import sphere

# This should be a matlab 6 compatible file (save -6 <file> <vars>
# in octave) as created by upsample_irs.m
# samples_to_keep
def load_irs_and_delaydiffs(filename = 'irs_and_delaydiffs_compensated_6.mat', samples_to_keep = 512):
	m = sp.io.loadmat(filename)['irs_and_delaydiffs']

	class irs_and_delaydiffs:
		upsampling = int(m[0][0]['upsampling'][0][0])

		diffs_left = m[0][0]['diffs_left']
		diffs_right = m[0][0]['diffs_right']

		irs_left = m[0][0]['irs_left'][:,:samples_to_keep * upsampling]
		irs_right = m[0][0]['irs_right'][:,:samples_to_keep * upsampling]

	return irs_and_delaydiffs

# }}}

# HRTF interpolation
# {{{

# Returns the delay compensated interpolated version of a signal equivalent to:
# (1-alpha) before + alpha after, with before and after referring to indices
# in the irs_and_delaydiffs database
# irs_and_delaydiffs: class as returned by load_irs_and_delaydiffs
# before, after: integers (indices for the HRTF database)
# alpha: float between 0 and 1
# In addition to the impulse responess, this returns the delay differences of the left and right channel,
# i.e. the delay difference between the 'before' HRTF and the interpolated HRTF.
# TODO maybe use one array with an extra dimension for left and right channels?
def delay_compensated_interpolation_with_delaydiff(irs_and_delaydiffs, before: int, after: int, alpha: float, return_upsampled = False):
	upsampling = irs_and_delaydiffs.upsampling

	# get the impulse responses of the 'before' sampling point
	# l_before = irs_and_delaydiffs.irs_left[before,:]
	# r_before = irs_and_delaydiffs.irs_right[before,:]

	# get the delay differences for both channels, convert them back to the upsampled domain
	delay_l = upsampling * irs_and_delaydiffs.diffs_left[before, after]
	delay_r = upsampling * irs_and_delaydiffs.diffs_right[before, after]

	# get the ir's of the 'after' sampling point, and remove the delay
	l_after_nodelay = delay_signal_float(irs_and_delaydiffs.irs_left[after,:], -delay_l)
	r_after_nodelay = delay_signal_float(irs_and_delaydiffs.irs_right[after,:], -delay_r)

	# interpolate the delay-free impulse responses
	l_interpolated_nodelay = (1-alpha) * irs_and_delaydiffs.irs_left[before,:] + alpha * l_after_nodelay
	r_interpolated_nodelay = (1-alpha) * irs_and_delaydiffs.irs_right[before,:] + alpha * r_after_nodelay

	# interpolate delays and add them back to the signals
	delay_l_interpolated = alpha * delay_l
	delay_r_interpolated = alpha * delay_r

	if return_upsampled:
		l_interpolated = delay_signal_float(l_interpolated_nodelay, delay_l_interpolated, 1);
		r_interpolated = delay_signal_float(r_interpolated_nodelay, delay_r_interpolated, 1);
	else:
		l_interpolated = delay_signal_float(l_interpolated_nodelay, delay_l_interpolated, upsampling);
		r_interpolated = delay_signal_float(r_interpolated_nodelay, delay_r_interpolated, upsampling);

	# TODO find a better way to stick two arrays together
	out_irs = np.concatenate([l_interpolated, r_interpolated]).reshape([2, l_interpolated.size])

	return (delay_l_interpolated / upsampling, delay_r_interpolated / upsampling, out_irs)

def delay_compensated_interpolation(irs_and_delaydiffs, before: int, after: int, alpha: float):
	(delay_l, delay_r, out_irs) = delay_compensated_interpolation_with_delaydiff(irs_and_delaydiffs, before, after, alpha)
	return out_irs


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

def interpolate_2d_deg(irs_and_delaydiffs, elev, azim):
	deg2rad = (2*np.pi) / 360
	return interpolate_2d(irs_and_delaydiffs, deg2rad * elev, deg2rad * azim)

def interpolate_2d(irs_and_delaydiffs, elev, azim):
	'''
	elev, azim in radians
	irs_and_delaydiffs as returned by load_irs_and_delaydiffs
	'''

	'''
	First attempt at 2d HRTF interpolation

	The idea is to use the already existing 1d interpolation to find the
	interpolated HRTFs for the given azimuth, at the next lower and higher
	elevations in the database. these two HRTFs can then be interpolated
	again to represent an elevation angle between the two adjacent ones.

	concerns/thoughts/ideas:
	- we need a way to get the final, interpolated delay difference from
	  the two calls of the 1d interpolation function and use them for the
	  final interpolation along the azim direction. this probably involves
	  a new, slightly different version of the function (although the old
	  one could easily be redefined in terms of the new one to avoid
	  redundancy).
	- this will be (at least) 3x slower than 1d interpolation, so using
	  this function incentivizes also implementing the make_signal_move
	  function in a more efficient manner, as outlined in a comment there.
	'''
	available_elevs = [-45,-30,-15,0,15,30,45,60,75,90]
	lower_elev = max([e for e in available_elevs if e < elev])
	higher_elev = min([e for e in available_elevs if e > elev])

	# get the adjacent indices and interpolation parameters
	(top_left, top_alpha, top_right) = sphere.azim_to_interpolation_params(higher_elev, azim)
	(bot_left, bot_alpha, bot_right) = sphere.azim_to_interpolation_params(lower_elev, azim)

	'''
	TODO maybe fall back to 2d interpolation if elev is in available_elevs?
	Although this would only be a null set among all possible (elev, azim)
	pairs, it is imaginable that lots of sound sources will have elev=0.
	'''

	'''
	TODO write the rest of the function :)
	- interpolate in one dimension to get the HRTFs for (elev, azim) =
	  (lower_elev, azim) or (higher_elev, azim)
	- somehow get the delay differences
	- interpolate between the two previously calculated HRTFs to get the
	  final HRTF for (elev, azim)
	'''



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
	sub-chunks, where each chunk has a newly computed delay compensated
	interpolated IR at the start and end, and within the chunk, the IR is
	simply interpolated linearly between the two actual interpolations for
	each sub-chunk. The chunks would have to be small enough that the delay
	difference between the two adjacent interpolated impulse response (at
	the start of one chunk and the next one) is much smaller (todo: how
	much?) than one sample.

	For example, we could use chunks of 400 and sub-chunks of 10 samples
	(400 samples = 9 ms, during which a sound source would have to travel
	at an angular velocity of (...) to exceed a difference in delay
	difference more than 1 sample)

	(TODO: actually calculate this and maybe implement a dynamic algorithm
	based on the difference quotient of index_function, assuming that
	index_function is sufficiently smooth)
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
			print(' {:.1f}%           '.format(100 * i / in_length), end='\r')
	print(' 100.0%')

	out_sig = np.concatenate([out_l, out_r]).reshape([2, out_l.size]).astype(np.float32).T
	return out_sig / np.max([out_sig.max(), -(out_sig.min())])
# }}}

def main():
	start = time.time()
	try:
		input_filename = sys.argv[1]
	except IndexError:
		printf('$1 empty - should be input file', file=sys.stderr)
		sys.exit(1)

	fs, y = wavfile.read(input_filename)
	y = y.astype(np.float32) / y.max()

	samples_to_keep = 120;
	T=2 # Period of signal moving around head
	chunksize = 50
	stereo_mode = False

	irs_and_delaydiffs = load_irs_and_delaydiffs('irs_and_delaydiffs_compensated_6.mat', samples_to_keep = samples_to_keep)

	if len(y.shape) == 2:
		if stereo_mode:
			print('The file \'{}\' is in stereo - enabling stereo mode!!! (this will take twice as long)'.format(input_filename))
			f_left = lambda t: (t % (T*fs)) * (24 / (T*fs)) + 73
			f_right = lambda t: (t % (T*fs)) * (24 / (T*fs)) + 73 + 12 # 12 indices = half a circle

			# Calculate the contributions from the left and right channels separately
			out_left = make_signal_move(y[:,0], chunksize, f_left, irs_and_delaydiffs)
			out_right = make_signal_move(y[:,1], chunksize, f_right, irs_and_delaydiffs)

			# Average them (both are already stereo)
			out_sig = 0.5 * (out_left + out_right)
		else:
			print('The file \'{}\' is in stereo, but stereo mode is off - converting to mono...'.format(input_filename))
			y_mono = 0.5 * (y[:,0] + y[:,1])
			out_sig = make_signal_move(y_mono, chunksize, lambda t: (t % (T*fs)) * (24 / (T*fs)) + 73, irs_and_delaydiffs).astype(np.float32);
	elif len(y.shape) == 1:
		# we have a mono signal
		out_sig = make_signal_move(y, chunksize, lambda t: (t % (T*fs)) * (24 / (T*fs)) + 73, irs_and_delaydiffs).astype(np.float32);
	else:
		printf('wrong input shape: {}'.format(y.shape), file=sys.stderr)
		sys.exit(1)

	out_filename = 'python-{}-{}-{}.wav'.format(
			input_filename.replace('.wav',''),
			chunksize,
			samples_to_keep);
	wavfile.write(out_filename, fs, out_sig.astype(np.float32))

	elapsed_time = time.time() - start;
	print("wrote to '{}' - took {:.2f} secs - {:.2f} faster than real time".format(
		out_filename,
		elapsed_time,
		(y.size / fs) / (elapsed_time)))

if __name__ == '__main__':
	main()
