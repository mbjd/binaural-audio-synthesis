#!/usr/bin/env python3

# Python rewrite of apply_hrtf.m

'''

This is the main file of my (Balduin Dettling, dbalduin (at)
student.ethz.ch) binaural audio generation project. The goal is to take an
audio file, along with a function that maps time to a point on the
unit sphere, and make another audio file that sounds as if the sound source
was moving around the listener according to the given function.

For this, I need measurements of HRTFs from different directions (1) (head
related transfer functions - the impulse responses of the human ears and
head). Luckily, that has already been done several times - in this project
I used a database of HRTF measurements spaced about 15 degrees.

The main challenge of the project then became interpolation, that is
creating a HRTF for an angle for which we don't have a measurement. The
simplest thing to do would be linearly interpolating between adjacent
HRTFs, which didn't work well because the difference in signal delay
sometimes causes unwanted constructive or destructive interference.

Another, slightly more advanced approach, which is now used in the project,
is what we call "delay compensated interpolation" (2). We first determine the
delay difference (3) between two signals between which we want to
interpolate. Then we remove the delay in the second signal (which is just a
shift in time), so that the signals have the same delay and delay
difference 0. We then linearly interpolate between the two signals,
according to an interpolation parameter alpha in [0,1]. To this
interpolated signal we add back an interpolated delay, which again is just
the linear interpolation between the delay differences of the two adjacent
measured impulse response.

Using this interpolation method, I then constructed a 2d interpolation
function, interpolate_2d, which is explained thoroughly in a comment in the
function itself.

Now we have a method to get a decent HRTF for any point on the sphere
(well, except if the elevation angle is below -45º, because the database
doesn't have measurements there). The rest of the challenge consists in
applying these HRTFs to a signal with a moving sound source. I experimented
with several methods, and have arrived at the abomination that is
make_signal_move_2d. It is also explained in a comment within the function
itself.

(1): If you want to do a slightly simpler project, consider doing away with
HRTFs and just try to recreate the amplitude and delay differences between
the ears from all possible angles.

(2): Implementation in delay_compensated_interpolation_with_delaydiff

(3): The delay difference of two signals is, loosely speaking, the distance
between the positions of the main peaks in the respective HRTF. The actual
definition is its implementation (Salamon would be proud of me), which is
the function delaydifference in ./upsample_irs.m, where we cross-correlate
the two impulse responses and basically return the position of the peak of
that signal in comparison to the middle of it (where the peak would be if
the two signals were the same)

'''

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
from mpl_toolkits.mplot3d import Axes3D

import sphere

# This should be a matlab 6 compatible file (save -6 <file> <vars>
# in octave, I haven't tested it in matlab itself) as created by upsample_irs.m
# samples_to_keep: length of the impulse response, the rest is truncated away
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
# (1-alpha) * before + alpha * after, with before and after referring to indices
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

	out_irs = np.vstack([l_interpolated, r_interpolated])

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
	available_elevs = np.deg2rad(np.array([-45,-30,-15,0,15,30,45,60,75,90]))

	try:
		lower_elev = max([e for e in available_elevs if e <= elev])
	except ValueError:
		lower_elev = -0.78539816339744828 # = deg2rad(-45)

	try:
		higher_elev = min([e for e in available_elevs if e >= elev])
	except ValueError:
		higher_elev = 1.5707963267948966 # = deg2rad(+90)

	assert higher_elev >= lower_elev, 'something\'s messed up'

	# get the adjacent indices and interpolation parameters
	(top_before, top_alpha, top_after) = sphere.azim_to_interpolation_params(higher_elev, azim)
	(bot_before, bot_alpha, bot_after) = sphere.azim_to_interpolation_params(lower_elev, azim)

	# calculate the 1d interpolated HRTFs at the next higher and lower elevations
	(delay_l_top, delay_r_top, hrtf_top) = delay_compensated_interpolation_with_delaydiff(irs_and_delaydiffs, top_before, top_after, top_alpha, return_upsampled=True)
	(delay_l_bot, delay_r_bot, hrtf_bot) = delay_compensated_interpolation_with_delaydiff(irs_and_delaydiffs, bot_before, bot_after, bot_alpha, return_upsampled=True)

	'''
	Interpolate vertically between the two horizontal interpolations.
	what follows is a version of delay_compensated_interpolation modified
	so heavily that I don't feel bad about writing it inline

	using a 'mesh rule'* for the delay difference function, we can derive:
	dd(top_interpolated, bot_interpolated) =
				top_alpha * dd(top_after, top_before)
				+ dd(top_before, bottom_left)
				+ bot_alpha * dd(bottom_left, bottom_right)

	* which is that the sum of delay differences in a loop should be zero,
	e.g. dd(a, b) + dd(b, c) + dd(c, a) = 0. I haven't proven this, but it
	trivially holds if the signals shifted delta functions, and it seems to
	work well enough for the signals we're interested in.
	Also btw: dd(a, b) = -dd(b, a) (proof left as exercise for reader)


	'''

	upsampling = irs_and_delaydiffs.upsampling

	delay_l = upsampling * (-delay_l_top
	+ irs_and_delaydiffs.diffs_left[top_before, bot_before]
	+ delay_l_bot)

	delay_r = upsampling * (-delay_r_top
	+ irs_and_delaydiffs.diffs_right[top_before, bot_before]
	+ delay_r_bot)

	# l_top = hrtf_top[0,:]
	# r_top = hrtf_top[1,:]

	l_bottom_nodelay = delay_signal_float(hrtf_bot[0,:], -delay_l)
	r_bottom_nodelay = delay_signal_float(hrtf_bot[1,:], -delay_r)

	# vertical interpolation parameter a ∈ [0,1]
	# a=0 -> only take bottom HRTF
	# a=1 -> only take top HRTF
	if higher_elev > lower_elev:
		a = (elev - lower_elev) / (higher_elev - lower_elev)
	else:
		a = 0
	assert 0 <= a <= 1, 'interpolation parameter somehow takes invalid value'

	l_interpolated_nodelay = (1-a) * l_bottom_nodelay + a * hrtf_top[0,:]
	r_interpolated_nodelay = (1-a) * r_bottom_nodelay + a * hrtf_top[1,:]

	# interpolate the delays
	delay_l_interpolated = (1-a) * delay_l
	delay_r_interpolated = (1-a) * delay_r

	# add back delays & downsample again
	l_interpolated = delay_signal_float(l_interpolated_nodelay, delay_l_interpolated, downsample=upsampling)
	r_interpolated = delay_signal_float(r_interpolated_nodelay, delay_r_interpolated, downsample=upsampling)

	out_irs = np.vstack([l_interpolated, r_interpolated])

	return out_irs

	'''
	TODO maybe fall back to 2d interpolation if elev is in available_elevs?
	Although this would only be a null set among all possible (elev, azim)
	pairs, it is imaginable that lots of sound sources will have elev=0.
	'''

	'''
	TODO write the rest of the function :)
	- interpolate in one dimension to get the HRTFs for (elev, azim) =
	  (lower_elev, azim) or (higher_elev, azim) - DONE
	- somehow get the delay differences - DONE
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
	print(' 100.0%      ')

	out_sig = np.vstack([out_l, out_r]).astype(np.float32).T

	m = np.max([out_sig.max(), -(out_sig.min())])
	if m > 1:
		out_sig /= m

	return out_sig


def make_signal_move_2d(in_signal, chunksize: int, subchunksize: int, elev_azim_function, irs_and_delaydiffs):
	'''
	This function makes the audio signal in_signal sound as if the sound
	source was moving according to elev_azim_function.

	First of all, the obvious thing to do would be:

		divide the input signal into chunks of size K, with each chunk starting at sample j = n*k
		initialize the output signal to zeros, with a length slightly longer than the input signal*
		for each chunk, do this:
			evaluate elev_azim_function at j, and calculate an interpolated HRTF at this (elev, azim) point
			convolve the chunk with the interpolated HRTF
			add the convolved chunk to the output signal at the right place (see: overlap-and-add)
		return the output signal

		* because convolving two signals a and b gives you a signal of length len(a) + len(b) - 1

	Now, K needs to be small (empirically, < 50-100, depending on how fast you move the
	sound source) for the signal to sound good and not have any 'clicks' from a sudden change of the
	HRTF. However, the smaller we choose K, the more often we will have to call the 2d interpolation
	function, which is quite expensive.

	To remedy this, we divide each chunk into "subchunks", and while we calculate a new HRTF for
	every chunk with our fancy 2d delay compensated interpolation function, we only linearly
	interpolate for each subchunk, which is much faster. Now, the subchunk size can be chosen quite
	small, like 16 or 32 samples, and the chunk size can be a bit larger than without subchunks,
	like 128-1024.  Again, this all depends on how quickly you move the sound source, the faster it
	moves, the lower you need to choose the chunk and subchunk sizes. (idea for the future:
	dynamically adjust the chunk and subchunk sizes depending on the current speed of the sound
	source)

	in_signal: input signal, ndarray of shape (1, N) = (N,)
	chunksize: Number of samples for which to use the same HRTF interpolation
	           (TODO: make this depend on the derivative of elev_azim_function)
	elev_azim_function: function of time (in samples) returning a radian (elev, azim) tuple
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
	# reallocates the upsampled IR's anyway which are way longer
	in_subchunk = np.zeros([subchunksize])
	out_subchunk_left = np.zeros([subchunksize + ir_length - 1])
	out_subchunk_right = np.zeros([subchunksize + ir_length - 1])
	out_indices = np.zeros([subchunksize + ir_length - 1], dtype=np.uint64)

	# here we later store the HRTF impulse responses
	ir_startchunk = np.zeros([2,ir_length])
	ir_endchunk = np.zeros([2,ir_length])
	ir_interpolated = np.zeros([2,ir_length]) #

	ir_endchunk = interpolate_2d(irs_and_delaydiffs, *(elev_azim_function(0)))
	for i in range(0, in_length, chunksize):

		# swap the support functions at the ends of the chunk
		ir_startchunk[:,:] = ir_endchunk[:,:]
		ir_endchunk[:,:] = interpolate_2d(irs_and_delaydiffs, *(elev_azim_function(i+chunksize)))

		for j in range(0, chunksize, subchunksize):
			in_subchunk[:] = in_signal[i+j:i+j+subchunksize];

			# linear interpolation between the two support functions, which are the 2d-interpolated
			# HRTFs at the start and end of the chunk
			alpha = j / chunksize # interpolation parameter in [0,1)
			ir_interpolated[:,:] = (1-alpha) * ir_startchunk[:,:] + alpha * ir_endchunk[:,:]

			out_subchunk_left[:]  = sp.signal.convolve(in_subchunk, ir_interpolated[0,:])
			out_subchunk_right[:] = sp.signal.convolve(in_subchunk, ir_interpolated[1,:])

			# sorry for complicated index math but it works
			# length of convolved subchunk   ->   ----------------------------
			out_indices[:] = np.arange(i+j, i+j + subchunksize + ir_length - 1)

			out_l[out_indices] += out_subchunk_left;
			out_r[out_indices] += out_subchunk_right;


		print(' {:.1f}%           '.format(100 * i / in_length), end='\r')
	print(' 100.0%      ')

	out_sig = np.vstack([out_l, out_r]).astype(np.float32).T

	m = np.max([out_sig.max(), -(out_sig.min())])
	if m > 1:
		out_sig /= m

	return out_sig
# }}}

# Display / Debugging / etc {{{
def imshow_interpolation(irs_and_delaydiffs, start, stop, steps, disp_upsample=4, new=True, func=None):
	'''
	steps: how many samples to calculate
	start, stop: tuples of (elev, azim) (degrees)
	'''

	if func:
		endtime = 8
		points = np.array([func(t) for t in np.linspace(0, 44100*endtime, steps)])
		# import pdb; pdb.set_trace()
		elevs = points[:,0]
		azims = points[:,1]
	else:
		# get the sampling points
		elevs = np.deg2rad(np.linspace(start[0], stop[0], steps))
		azims = np.deg2rad(np.linspace(start[1], stop[1], steps))

	# plot elev/azim points in 2d {{{
	pl.plot(elevs)
	pl.plot(azims)
	pl.show()
	# }}}

	# plot points in 3d {{{
	fig = pl.figure()
	ax = fig.add_subplot(111, projection='3d')
	cart_samplepoints = sphere.get_cartesian_samplepoints()
	(xs, ys, zs) = (cart_samplepoints[:,0], cart_samplepoints[:,1], cart_samplepoints[:,2])
	ax.scatter(xs, ys, zs, c='blue')

	# column 0 = x coordinate
	xs = -np.sin(azims) * np.cos(elevs);
	# column 1 = y coordinate
	ys = np.cos(azims) * np.cos(elevs);
	# column 2 = z coordinate
	zs = np.sin(elevs)

	ax.scatter(xs, ys, zs, c='red')
	pl.show()
	# }}}

	ir_length = int(0.5 + irs_and_delaydiffs.irs_left.shape[1] / irs_and_delaydiffs.upsampling)
	irs_l = np.zeros([steps, ir_length * disp_upsample])
	irs_r = np.zeros([steps, ir_length * disp_upsample])

	# fill the matrix, each line is one HRTF
	for i in range(steps):
		print(' {:.2f}%                  '.format(100 * i/steps), end='\r')
		if new:
			irs = interpolate_2d(irs_and_delaydiffs, elevs[i], azims[i])
		else:
			irs = delay_compensated_interpolation_easy(irs_and_delaydiffs, 73 + (24/360)*azims[i])
		irs = scipy.signal.resample(irs, disp_upsample * ir_length, axis=1)
		irs_l[i,:] = irs[0,:]
		irs_r[i,:] = irs[1,:]
	print(' 100%       ')

	# show the image of HRTFs {{{
	fig, (ax1, ax2) = pl.subplots(1,2, sharex=True, sharey=True)

	ax1.imshow(irs_l, aspect='auto', extent=(0, ir_length, steps, 0))
	ax1.set_title('left HRTFs')
	ax1.set_xlabel('Samples')
	ax1.set_ylabel('Steps')

	ax2.imshow(irs_r, aspect='auto', extent=(0, ir_length, steps, 0))
	ax2.set_title('right HRTFs')
	ax2.set_xlabel('Samples')
	ax2.set_ylabel('Steps')

	pl.show()
	# }}}


# }}}

def main():

	'''
	~~~ Main function ~~~

	- get a filename from the first command line argument
	- read audio
	'''

	try:
		input_filename = sys.argv[1]
	except IndexError:
		print('argv[1] empty - should be input file', file=sys.stderr)
		sys.exit(1)

	fs, y = wavfile.read(input_filename)
	y = y.astype(np.float32) / y.max()

	# functions returning an (elev, azim) tuple (in radians)
	T=4 # Period of signal moving around head
	A = 1
	k = 2*np.pi / (T*fs)
	circle_front = lambda t: (A*np.sin(k*t), A*np.cos(k*t))

	circle_horizontal = lambda t: (0, (k*t) % (2*np.pi))
	circle_askew = lambda t: ((np.pi/4)*np.cos(k*t), (k*t) % (2*np.pi))
	halfcircle_vertical = lambda t: ((np.pi/2) * (1 - 1.5*np.abs(np.cos(k*t))), (np.pi/2) * np.sign(np.cos(k*t)))
	passing = lambda t: (0, np.arctan(12 * np.cos(2*k*t)))

	# in seconds
	length = 30
	turns = 15
	spiral = lambda t: ((-np.pi/4) + (3*np.pi/4) * (t/(fs*length)), 2*np.pi*t*turns/(fs*length))

	samples_to_keep = 100;
	stereo_mode = False
	chunksize = 512
	subchunksize = 32

	start = time.time()

	irs_and_delaydiffs = load_irs_and_delaydiffs('irs_and_delaydiffs_compensated_6.mat', samples_to_keep = samples_to_keep)

	if len(y.shape) == 2 and y.shape[1] == 2:
		if stereo_mode:
			# disclaimer - this is old, may not work anymore {{{
			# apply the (constant) left/right HRTFs to the left and right channels, maybe this will make it sound more realistic?
			left = lambda t: (0, ((2*np.pi/(8*fs)+np.pi/2) % 2*np.pi))
			right = lambda t: (0, ((2*np.pi/(8*fs)+3*np.pi/2) % 2*np.pi))

			left_out = make_signal_move_2d(y[:,0], chunksize, subchunksize, left, irs_and_delaydiffs).astype(np.float32)
			right_out = make_signal_move_2d(y[:,1], chunksize, subchunksize, right, irs_and_delaydiffs).astype(np.float32)

			out_sig = 0.5 * (left_out + right_out)
			out_filename = '{}-binaural-stereo.wav'.format(input_filename.replace('.wav',''))
			wavfile.write(out_filename, fs, out_sig.astype(np.float32))

			elapsed_time = time.time() - start;
			print("wrote to '{}' - took {:.2f} secs - {:.2f}x as fast as real time".format(
				out_filename,
				elapsed_time,
				(y.size / fs) / (elapsed_time)))
			return
			# }}}

		else:
			# Convert the signal to mono
			y = 0.5 * y[:,0] + 0.5 * y[:,1]
			assert len(y.shape) == 1

	out_sig = make_signal_move_2d(y, chunksize, subchunksize, passing, irs_and_delaydiffs).astype(np.float32)

	# chunk size, subchunk size, length of impulse response
	out_filename = '{}-c{}-s{}-l{}.wav'.format(
			input_filename.replace('.wav',''),
			chunksize, subchunksize,
			samples_to_keep);
	wavfile.write(out_filename, fs, out_sig.astype(np.float32))

	elapsed_time = time.time() - start;
	print("wrote to '{}' - took {:.2f} secs - {:.2f}x as fast as real time".format(
		out_filename,
		elapsed_time,
		(y.size / fs) / (elapsed_time)))

if __name__ == '__main__':
	main()
