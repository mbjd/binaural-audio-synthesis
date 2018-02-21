% experiments with applying a HRTF to a (mono) audio signal

% DISCLAIMER
% This script has been abandoned, and newer developments have been
% made in the python version apply_hrtf.py.

% TODO
% update delay_compensated_interpolation_efficient and delay_signal_int to
% be able to delay a signal by a float by linearly interpolating between the
% signals shifted by the two adjacent integers. this will not introduce
% significant high frequency noise (which would later be aliased when downsampling),
% as long as the bandwidth of the signal is much lower than the nyquist frequency,
% fs/2. this is the case as long as the upsampling factor used in upsample_irs
% is larger than 1.

% INPUT ETC.
% {{{
pkg load signal

% read an input signal
% [input_signal FS] = audioread('short-singleschnips-44100.wav');
[input_signal FS] = audioread('whitenoise.wav');
% [input_signal FS] = audioread('squarewave.wav');

% which HRTF to use:                         --> ==== <--
load recherche.ircam.fr/COMPENSATED/MAT/HRIR/IRC_1032_C_HRIR.mat
% load recherche.ircam.fr/RAW/MAT/HRIR/IRC_1026_R_HRIR.mat
% http://recherche.ircam.fr/equipes/salles/listen/download.html

% has to be generated by upsample_irs first
load irs_and_delaydiffs_compensated.mat

% TODO handle different sampling rates
assert (l_eq_hrir_S.sampling_hz == FS);
assert (r_eq_hrir_S.sampling_hz == FS);
% }}}

% BASIC HRTF STUFF
% {{{

% TODO separate the functions for finding the HRTF for an angle
% and convolving it with the signal, to make it easier to interpolate later

% elev, azim: the position of the simulated sound source is relative to the listener's head
% input_signal: the input signal as given by audioread
% l_eq_hrir_S, r_eq_hrir_S: the two structs contained in the recherche.ircam.fr mat files
function out_signal = apply_hrtf_to_signal(input_signal, elev, azim, l_eq_hrir_S, r_eq_hrir_S)

	% find the best approximation for (elev, azim) within the HRTF database
	% {{{
	% TODO handle wrapping around (0 = 2pi = 360 deg)
	% compute the elev and azim distances to the desired value
	elev_distances = l_eq_hrir_S.elev_v - (elev * ones(size(l_eq_hrir_S.elev_v)));
	azim_distances = l_eq_hrir_S.azim_v - (azim * ones(size(l_eq_hrir_S.azim_v)));

	% l2 norm of the distance vector in the (elev, azim) plane
	elev_azim_distances = sqrt(elev_distances .^ 2 + azim_distances .^ 2);

	% find the minimal one
	% the unit of min_elev_azim_dist is degrees in theory but
	% this is only accurate near the equator
	[min_elev_azim_dist, min_elev_azim_index] = min(elev_azim_distances);

	% disp(strcat('desired: (elev, azim) = (', num2str(elev), ',', num2str(azim), ')'))
	% disp(strcat('actual:  (elev, azim) = (', num2str(l_eq_hrir_S.elev_v(min_elev_azim_index)), ',', num2str(l_eq_hrir_S.azim_v(min_elev_azim_index)), ')'))
	% }}}

	left_impulse_response = l_eq_hrir_S.content_m(min_elev_azim_index,:);
	right_impulse_response = r_eq_hrir_S.content_m(min_elev_azim_index,:);

	out_signal_l = fftconv(left_impulse_response, input_signal);
	out_signal_r = fftconv(right_impulse_response, input_signal);

	out_signal = [out_signal_l out_signal_r];
endfunction

% }}}

% INTERPOLATION WITHOUT DELAY COMPENSATION
% {{{
% get the impulse response between the two other ones
% out_ir = (1-a) * irs(index_1) + a * irs(index_2)
function out_irs = interpolate_impulse_response_simple_manual(l_eq_hrir_S, r_eq_hrir_S, index_1, index_2, a)
	assert (0 <= a <= 1);
	out_left  = (1 - a) * l_eq_hrir_S.content_m(index_1,:) + a * l_eq_hrir_S.content_m(index_2,:);
	out_right = (1 - a) * r_eq_hrir_S.content_m(index_1,:) + a * r_eq_hrir_S.content_m(index_2,:);
	out_irs = [out_left' out_right'];
endfunction

% Same as above but different interface
% TODO find out how function overloading works in octave
% simply specify a float as HRTF index (as in the database) and it will return the correct interpolation
function out_irs = interpolate_impulse_response_simple(l_eq_hrir_S, r_eq_hrir_S, continuous_index)
	before = floor(continuous_index);
	after = ceil(continuous_index);
	a = continuous_index - before;
	out_irs = interpolate_impulse_response_simple_manual(l_eq_hrir_S, r_eq_hrir_S, before, after, a);
endfunction
% }}}

% INTERPOLATION WITH DELAY COMPENSATION
% {{{

% instead of just linearly interpolating between two IRs, do this:
% - find the time delay difference of the signals
% - shift one signal so the delay becomes 0
% - linearly interpolate between these two delay-normalised signals
% - add back a delay which is the interpolation between the two delays
% TODO: to have this run somewhat efficiently it would probably be smart to
%       pre-calculate upsampled versions of all the impulse responses, as these
%       signals are needed to find out the delay and to shift the signals around
%       by less than 1 sample. (possible solution, upsample 5x - 10x and linearly
%       interpolate the rest, TODO test if this sounds good)
%       OR we do this and additionally precalculate the delays to avoid cross
%       correlation altogether (at runtime)
% TODO: use the float delay by linearly interpolating between the two adjacent delayed versions
function out_sig = delay_compensated_interpolation_manual(signal_a, signal_b, upsampling, a)
	assert (0 <= a <= 1, 'interpolation parameter a needs to be in [0, 1]');

	% delay = delaydifference(signal_a, signal_b, upsampling);
	a_upsampled = resample(signal_a, upsampling, 1);
	b_upsampled = resample(signal_b, upsampling, 1);

	% delay in samples of b in relation to a
	delay = delaydifference(a_upsampled, b_upsampled, 1);

	% TODO consider the actual delay by linearly interpolating
	% between shifted signal versions for accuracy < 1 sample
	delay_int = floor(delay);
	b_upsampled_without_delay = delay_signal_int(b_upsampled, -delay_int);

	% plot([b_upsampled_without_delay; a_upsampled]');
	assert(abs(delaydifference(b_upsampled_without_delay, a_upsampled, 1)) < 1, 'delay is still bigger than 1 sample even though it should have been removed');

	linear_interpolation_without_delay = (1-a) * a_upsampled + a * b_upsampled_without_delay;
	interpolated_delay = floor(a * delay);

	out_sig = resample(delay_signal_int(linear_interpolation_without_delay, interpolated_delay), 1, upsampling);
endfunction

% calculate the delay compensated interpolated impulse response more efficiently.
% irs_and_delaydiffs is a struct as calculated in upsample_irs.m
function out_sig = delay_compensated_interpolation_efficient(irs_and_delaydiffs, continuous_index, float_delay)
	upsampling = irs_and_delaydiffs.upsampling;

	% convert from float index to two int indices and interpolation parameter a
	before = floor(continuous_index);
	after = ceil(continuous_index);
	a = continuous_index - before;

	% TODO remove this later
	if (after == 97) after = 73; endif

	% look up delay difference in upsampled samples
	delay_left = upsampling * irs_and_delaydiffs.diffs_left(before, after);
	delay_right = upsampling * irs_and_delaydiffs.diffs_right(before, after);

	% get the impulse responses
	left_before  = irs_and_delaydiffs.irs_left(before,:);
	right_before = irs_and_delaydiffs.irs_right(before,:);

	if (float_delay)
		left_after_nodelay = delay_signal_float(irs_and_delaydiffs.irs_left(after,:), -delay_left);
		right_after_nodelay = delay_signal_float(irs_and_delaydiffs.irs_right(after,:), -delay_right);
	else
		left_after_nodelay = delay_signal_int(irs_and_delaydiffs.irs_left(after,:), -floor(delay_left));
		right_after_nodelay = delay_signal_int(irs_and_delaydiffs.irs_right(after,:), -floor(delay_right));
	endif

	% interpolate the impulse responses
	% TODO maybe try to do the downsampling earlier for efficiency
	left_interpolated_nodelay = (1-a) * left_before + a * left_after_nodelay;
	right_interpolated_nodelay = (1-a) * right_before + a * right_after_nodelay;

	% interpolate the delays and add them back to the signals
	left_delay_interpolated = a * delay_left;
	right_delay_interpolated = a * delay_right;

	% add interpolated delay & downsample
	% TODO instead of resampling just cut the samples out that we need since there are no higher frequencies anyway
	if (float_delay)
		left_interpolated = delay_signal_float(left_interpolated_nodelay, left_delay_interpolated)(1:upsampling:length(left_interpolated_nodelay));
		right_interpolated = delay_signal_float(right_interpolated_nodelay, right_delay_interpolated)(1:upsampling:length(left_interpolated_nodelay));
		% left_interpolated = resample(delay_signal_float(left_interpolated_nodelay, left_delay_interpolated), 1, upsampling);
		% right_interpolated = resample(delay_signal_float(right_interpolated_nodelay, right_delay_interpolated), 1, upsampling);
	else
		left_interpolated = resample(delay_signal_int(left_interpolated_nodelay, floor(left_delay_interpolated)), 1, upsampling);
		right_interpolated = resample(delay_signal_int(right_interpolated_nodelay, floor(right_delay_interpolated)), 1, upsampling);
	endif

	out_sig = [left_interpolated' right_interpolated'];
endfunction

% Returns the left and right impulse response interpolated with delay compensation
function out_irs = delay_compensated_interpolation(l_eq_hrir_S, r_eq_hrir_S, continuous_index)
	upsampling = 10;

	% convert from float index to two int indices and interpolation parameter a
	before = floor(continuous_index);
	after = ceil(continuous_index);
	a = continuous_index - before;

	left = delay_compensated_interpolation_manual(l_eq_hrir_S.content_m(before,:), l_eq_hrir_S.content_m(after,:), upsampling, a);
	right = delay_compensated_interpolation_manual(r_eq_hrir_S.content_m(before,:), r_eq_hrir_S.content_m(after,:), upsampling, a);
	out_irs = [left' right'];
endfunction

% delays a signal by a given number of samples
function out_sig = delay_signal_int(in_sig, samples)
	assert (size(in_sig)(1) == 1, 'in_sig must be a row vector');
	assert(floor(samples) == samples, 'delay_signal_int: can only delay by integer number of samples');

	len = length(in_sig);

	if (samples > 0)
		% delay the signal by samples
		out_sig = [zeros([1 samples]) in_sig(1:len-samples)];
	elseif (samples < 0)
		% advance the signal by -samples
		samples = -samples;
		out_sig = [in_sig(samples+1:len) zeros([1 samples])];
	else
		out_sig = in_sig;
	endif
endfunction

% delays a signal by a given number of samples
function out_sig = delay_signal_float(in_sig, samples)
	assert (size(in_sig)(1) == 1, 'in_sig must be a row vector');

	before = floor(samples);
	after = ceil(samples);
	a = samples - before;

	delayed_before = delay_signal_int(in_sig, before);
	delayed_after = delay_signal_int(in_sig, after);

	out_sig = (1-a) * delayed_before + a * delayed_after;
endfunction

% Finds the delay difference between two HRTFs a and b
% For decent results choose two HRTFs that are 'close' to each other
% returns the delay between the responses (delay > 0 if b comes after a)
% TODO find out why this works
% TODO try replace these expensive calculations with a simple model of the delay or a LUT
function diff = delaydifference(signal_a, signal_b, upsampling)
	% business logic
	assert (length(signal_a) == length(signal_b))
	signallength = length(signal_a);

	% autocorrelate the signal and upsample
	autocorr_upsampled = resample(fftconv(fliplr(flipud(signal_a)), signal_b), upsampling, 1);

	% find the peak
	[value, peak_index] = max(autocorr_upsampled);
	peak_index += parabolic_interpolation(autocorr_upsampled(peak_index-1:peak_index+1)) - 1;

	% convert back to non-upsampled samples
	peak_index /= upsampling;

	% shift the peak so the middle (zero delay) is in the middle of the signal (2*len - 1)
	diff = peak_index - (signallength - 1);
endfunction

% finds the three coefficients (a,b,c) of the parabola f(x) = ax^2 + bx + c
% which fits the three points contained in vec, such that:
% f(-1) = vec(1)
% f(0) = vec(2)
% f(+1) = vec(3)
% and then returns the position of the peak relative to the middle value
% vec(2)
% CAUTION: if the 3 points all lie on a line we will divide by 0, but if
% the middle point is a strict maximum this will not happen (and the three
% points all having the same value is unlikely enough to ignore)
function peak = parabolic_interpolation(vec)
	% check the input
	assert (length(vec) == 3);
	[_, index] = max(vec);
	assert (index == 2);

	% proof left to the reader
	c = vec(2);
	a = 0.5 * (vec(1) + vec(3) - 2 * c);
	b = 0.5 * (vec(3) - vec(1));
	assert (a != 0);
	peak = -b/(2*a);

endfunction

% }}}

% DISPLAY FUNCTIONS
% {{{
% make a plot of a series of HRTFs in the time domain
% x-axis: time
% y-axis: azimuth angle of HRTF
% z-axis: amplitude
function imshow_interpolation(l_eq_hrir_S, r_eq_hrir_S, irs_and_delaydiffs, first, last, steps)
	% samples to show from the impulse response
	ir_length = 100;

	% upsample the impulse responses before displaying by this factor
	display_upsampling = 4;

	% simple linear interpolation
	image_left  = [];
	image_right = [];
	for index=linspace(first, last, steps)
		disp(index);
		fflush(stdout);
		impulse_resps = interpolate_impulse_response_simple(l_eq_hrir_S, r_eq_hrir_S, index);
		ir_left = resample(impulse_resps(1:ir_length, 1), display_upsampling, 1);
		ir_right = resample(impulse_resps(1:ir_length, 2), display_upsampling, 1);
		image_left = [image_left; ir_left'];
		image_right = [image_right; ir_right'];
	endfor
	subplot(321)
	axis('tight', [0 ir_length 0 360]);
	imagesc(image_left, [-1 1]);
	subplot(322)
	axis('tight', [0 ir_length 0 360]);
	imagesc(image_right, [-1 1]);

	% delay compensated interpolation with precalculated upsamplings and delays, with int delay
	image_left  = [];
	image_right = [];
	for index=linspace(first, last, steps)
		disp(index);
		fflush(stdout);
		impulse_resps = delay_compensated_interpolation_efficient(irs_and_delaydiffs, index, false);
		ir_left = resample(impulse_resps(1:ir_length, 1), display_upsampling, 1);
		ir_right = resample(impulse_resps(1:ir_length, 2), display_upsampling, 1);
		image_left = [image_left; ir_left'];
		image_right = [image_right; ir_right'];
	endfor
	subplot(323)
	axis('tight', [0 ir_length 0 360]);
	imagesc(image_left, [-1 1]);
	subplot(324)
	axis('tight', [0 ir_length 0 360]);
	imagesc(image_right, [-1 1]);


	% delay compensated interpolation with precalculated upsamplings and delays, with float delay
	image_left  = [];
	image_right = [];
	for index=linspace(first, last, steps)
		disp(index);
		fflush(stdout);
		impulse_resps = delay_compensated_interpolation_efficient(irs_and_delaydiffs, index, true);
		ir_left = resample(impulse_resps(1:ir_length, 1), display_upsampling, 1);
		ir_right = resample(impulse_resps(1:ir_length, 2), display_upsampling, 1);
		image_left = [image_left; ir_left'];
		image_right = [image_right; ir_right'];
	endfor
	subplot(325)
	axis('tight', [0 ir_length 0 360]);
	imagesc(image_left, [-1 1]);
	subplot(326)
	axis('tight', [0 ir_length 0 360]);
	imagesc(image_right, [-1 1]);
endfunction


function imshow_interpolation_dft(l_eq_hrir_S, r_eq_hrir_S, irs_and_delaydiffs, first, last, steps)
	% simple linear interpolation
	image_left  = [];
	image_right = [];
	for index=linspace(first, last, steps)
		disp(index);
		fflush(stdout);
		impulse_resps = interpolate_impulse_response_simple(l_eq_hrir_S, r_eq_hrir_S, index);
		impulse_resps = interpolate_impulse_response_simple(l_eq_hrir_S, r_eq_hrir_S, index);
		ir_left = abs(fft(impulse_resps(:, 1)))(1:256);
		ir_right = abs(fft(impulse_resps(:, 2)))(1:256);
		image_left = [image_left; ir_left'];
		image_right = [image_right; ir_right'];
	endfor
	subplot(321)
	axis('tight');
	imagesc(image_left, [-1 1]);
	subplot(322)
	axis('tight');
	imagesc(image_right, [-1 1]);

	% fancy delay compensated interpolation
	image_left  = [];
	image_right = [];
	for index=linspace(first, last, steps)
		disp(index);
		fflush(stdout);
		impulse_resps = delay_compensated_interpolation(l_eq_hrir_S, r_eq_hrir_S, index);
		ir_left = abs(fft(impulse_resps(:, 1)))(1:256);
		ir_right = abs(fft(impulse_resps(:, 2)))(1:256);
		image_left = [image_left; ir_left'];
		image_right = [image_right; ir_right'];
	endfor
	subplot(323)
	axis('tight');
	imagesc(image_left, [-1 1]);
	subplot(324)
	axis('tight');
	imagesc(image_right, [-1 1]);


	% delay compensated interpolation with precalculated upsamplings and delays
	image_left  = [];
	image_right = [];
	for index=linspace(first, last, steps)
		disp(index);
		fflush(stdout);
		impulse_resps = delay_compensated_interpolation_efficient(irs_and_delaydiffs, index);
		ir_left = abs(fft(impulse_resps(:, 1)))(1:256);
		ir_right = abs(fft(impulse_resps(:, 2)))(1:256);
		image_left = [image_left; ir_left'];
		image_right = [image_right; ir_right'];
	endfor
	subplot(325)
	axis('tight');
	imagesc(image_left, [-1 1]);
	subplot(326)
	axis('tight');
	imagesc(image_right, [-1 1]);
endfunction


% }}}

% TEST SIGNAL GENERATOR FUNCTIONS
% {{{

% construct a signal that repeats in_sig but makes it rotate around the listener's head
% this version doesn't wait for one signal to completely fade out before the next one starts
% assumption: input signal is a column vector, i.e. size(in_sig) == [.., 0]
function out_signal = continuous_circle_no_interpolation(in_sig, elev, azim_start_idx, azim_end_idx, l_eq_hrir_S, r_eq_hrir_S)
	ir_length = size(l_eq_hrir_S.content_m)(2);
	input_length = size(in_sig)(1);

	% output signal that fits all the convolved input signals + (length(impulse response) - 1)
	out_l = zeros([(azim_end_idx - azim_start_idx + 1) * input_length + (ir_length - 1), 1]);
	out_r = out_l;

	for azim_idx=(azim_start_idx:azim_end_idx)
		% index to start writing to the output signal
		i = (azim_idx - azim_start_idx) * input_length + 1;
		out_sig_indices = (i:i+(ir_length + input_length - 2))';
		out_l(out_sig_indices) = out_l(out_sig_indices) + fftconv(in_sig, l_eq_hrir_S.content_m(azim_idx,:)');
		out_r(out_sig_indices) = out_r(out_sig_indices) + fftconv(in_sig, r_eq_hrir_S.content_m(azim_idx,:)');
	endfor
	out_signal = [out_l out_r];
endfunction

% assumption: input signal is a column vector, i.e. size(in_sig) == [.., 0]
% in_sig has to be as long as the whole output signal this time, so the output
% doesn't sound bad if the chunks where one HRTF is applied are very small
% TODO find out how short signals need to be for conv to be faster than fftconv
function out_signal = continuous_circle_with_interpolation(in_sig, azim_start_idx, azim_end_idx, azim_steps, l_eq_hrir_S, r_eq_hrir_S)
	ir_length = size(l_eq_hrir_S.content_m)(2);
	input_length = size(in_sig)(1);

	% output signal that fits all the convolved input signals + (length(impulse response) - 1)
	out_l = zeros([input_length + ir_length - 1, 1]);
	out_r = out_l;

	% how long each impulse response is used for
	steplength_samples = floor((input_length - ir_length + 1) / azim_steps);

	i=0;
	for azim_idx=linspace(azim_start_idx, azim_end_idx, azim_steps)
		irs_interpolated = interpolate_impulse_response_simple(l_eq_hrir_S, r_eq_hrir_S, azim_idx);

		% indices to write to the output signal
		out_sig_indices = (1 + i * steplength_samples) + (1:steplength_samples + ir_length - 1);

		in_sig_indices = out_sig_indices(1:steplength_samples);
		in_sig_slice = in_sig(in_sig_indices);

		out_l(out_sig_indices) = out_l(out_sig_indices) + fftconv(in_sig_slice, irs_interpolated(:,1));
		out_r(out_sig_indices) = out_r(out_sig_indices) + fftconv(in_sig_slice, irs_interpolated(:,2));
		i += 1;
	endfor
	out_signal = [out_l out_r];
endfunction


function out_signal = make_signal_move(in_sig, chunksize, index_function, irs_and_delaydiffs)
	%{
	in_sig is a signal (nx1 column vector)
	chunksize: integer specifying for how many samples to use the same impulse response
	index_function: a function of time (in samples) specifying the continuous index of the impulse response for each point in time
	example:
	index_function = (@(t) t * (23/(input_signal_length_in_sec * 44100)) + 73) % for a full circle at 0 elevation
	%}

	assert(size(in_sig)(2) == 1, 'in_sig is not a column vector');
	ir_length = size(irs_and_delaydiffs.irs_left)(2) / irs_and_delaydiffs.upsampling;

	% output length = next bigger integer multiple of chunk size
	in_length = ceil(length(in_sig) / chunksize) * chunksize;
	% fill the input signal with 0's to make it equally long
	in_sig = [in_sig; zeros([in_length - length(in_sig) 1])];
	assert(in_length == length(in_sig), 'this should not have happened');
	assert(in_length == ceil(length(in_sig) / chunksize) * chunksize);

	out_length = in_length + ir_length - 1;

	out_l = zeros([out_length 1]);
	out_r = zeros([out_length 1]);

	% If chunksize is small, conv is faster
	if (chunksize < 450)
		conv_func = (@(a,b) conv(a,b));
	else
		conv_func = (@(a,b) fftconv(a,b));
	endif

	for i=(1:chunksize:in_length)
		% get the interpolated impulse response and the input signal
		irs_interpolated = delay_compensated_interpolation_efficient(irs_and_delaydiffs, index_function(i), true);
		in_chunk = in_sig(i:i+chunksize-1);

		% convolve for both channels
		out_chunk_left = conv_func(in_chunk, irs_interpolated(:,1));
		out_chunk_right = conv_func(in_chunk, irs_interpolated(:,2));

		% make sure the indices aren't messed up
		out_chunk_indices = (i:i+chunksize+ir_length-2);
		assert(length(out_chunk_indices) == length(out_chunk_left), 'error in calculation of indices');

		% write to the output signal
		out_l(out_chunk_indices) = out_l(out_chunk_indices) + out_chunk_left;
		out_r(out_chunk_indices) = out_r(out_chunk_indices) + out_chunk_right;

		printf('%.2f%%       \r', (100 * i / in_length)); fflush(stdout);
	endfor
	printf('\n');

	out_signal = [out_l out_r];
endfunction

% make a short sample rotate around the listener's head
function out_signal = circle(input_signal, elev, azim_step, l_eq_hrir_S, r_eq_hrir_S)
	degrees = 0;
	out_signal = [];
	while (degrees < 360)
		out_signal = [out_signal; apply_hrtf_to_signal(input_signal, 0, degrees, l_eq_hrir_S, r_eq_hrir_S)];
		degrees += azim_step;
		% disp(degrees);
	endwhile
	out_signal = out_signal ./ (2 * norm(out_signal, inf));
endfunction

% }}}

% filename = 'binauraltest.wav';
% audiowrite(filename, circle(input_signal, 0, 15, l_eq_hrir_S, r_eq_hrir_S), 44100);
% audiowrite(filename, continuous_circle_with_interpolation(rand([44100 * 2, 1])-0.5, 73, 96, 2000, l_eq_hrir_S, r_eq_hrir_S), 44100);
% disp(strcat('wrote:', filename));

% n=4;
% f = (@(t) mod(t, n*44100) * (24 / (n * 44100)) + 73);
% out = make_signal_move(audioread('netzerk.wav'), 20, f, irs_and_delaydiffs);
% audiowrite('test.wav', out, 44100);
