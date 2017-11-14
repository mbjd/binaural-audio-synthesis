% experiments with applying a HRTF to a (mono) audio signal

pkg load signal

% read an input signal
% [input_signal FS] = audioread('short-singleschnips-44100.wav');
[input_signal FS] = audioread('whitenoise.wav');
% [input_signal FS] = audioread('squarewave.wav');

% which HRTF to use:                         --> ==== <--
load recherche.ircam.fr/COMPENSATED/MAT/HRIR/IRC_1032_C_HRIR.mat
% load recherche.ircam.fr/RAW/MAT/HRIR/IRC_1026_R_HRIR.mat

% http://recherche.ircam.fr/equipes/salles/listen/download.html

% TODO handle different sampling rates
assert (l_eq_hrir_S.sampling_hz == FS);
assert (r_eq_hrir_S.sampling_hz == FS);

% plot HRTFs successively in the time domain
function plot_hrtf_impulse(l_eq_hrir_S, r_eq_hrir_S, start_index, end_index, pause_s)
	for i=(start_index:end_index)
		disp(strcat(
			'azim= ', num2str(l_eq_hrir_S.azim_v(i)),
			', elev=', num2str(l_eq_hrir_S.elev_v(i))
		));
		fflush(stdout);
		responses = [[l_eq_hrir_S.content_m(i,:)' + 1], r_eq_hrir_S.content_m(i,:)'];
		plot([[2.5 -1.5]; responses]);
		pause(pause_s)
	endfor
endfunction

% plot HRTFs successively in the frequency domain
function plot_hrtf_freq(l_eq_hrir_S, r_eq_hrir_S, start_index, end_index, pause_s)
	for i=(start_index:end_index)
		disp(strcat(
			'azim= ', num2str(l_eq_hrir_S.azim_v(i)),
			', elev=', num2str(l_eq_hrir_S.elev_v(i))
		));
		fflush(stdout);
		responses = log(abs([[fft(l_eq_hrir_S.content_m(i,:)')], fft(r_eq_hrir_S.content_m(i,:)')])((1:256),:));
		plot([responses]);
		pause(pause_s)
	endfor
endfunction


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

% return a vector containing norm(left, p) / norm(right, p) for every impulse response
% between the two indices
function v = amplitudedifference(start_index, end_index, l_eq_hrir_S, r_eq_hrir_S, p)
	v=[];
	for i=(start_index:end_index)
		v = [v; norm(l_eq_hrir_S.content_m(i,:), p) / norm(r_eq_hrir_S.content_m(i,:), p)];
	endfor
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


% TODO finish this
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
function out_irs = interpolate_impulse_response_delay_manual(l_eq_hrir_S, r_eq_hrir_S, index_1, index_2, a)
	delay_diff_l = delaydifference(l_eq_hrir_S.content_m(index_1,:), l_eq_hrir_S.content_m(index_2,:), 4);
	delay_diff_r = delaydifference(r_eq_hrir_S.content_m(index_1,:), r_eq_hrir_S.content_m(index_2,:), 4);

	interpolated_delay_l = a * delay_diff_l;
	interporated_deray_r = a * deray_diff_r;
endfunction

% Finds the delay difference between two HRTFs a and b
% For decent results choose two HRTFs that are 'close' to each other
% returns the delay between the responses (delay > 0 if a comes after b)
% TODO find out why this works
% TODO try replace these expensive calculations with a simple model of the delay or a LUT
function diff = delaydifference(signal_a, signal_b, upsampling)
	% business logic
	assert (length(signal_a) == length(signal_b))
	signallength = length(signal_a);

	% autocorrelate the signal and upsample
	autocorr_upsampled = resample(fftconv(signal_a, fliplr(flipud(signal_b))), upsampling, 1);

	% find the peak
	[value, peak_index] = max(autocorr_upsampled);
	peak_index += parabolic_interpolation(autocorr_upsampled(peak_index-1:peak_index+1)) - 1;

	% convert back to non-upsampled samples
	peak_index /= upsampling;

	% shift the peak so the middle (zero delay) is in the middle of the signal (2*len - 1)
	diff = peak_index - (signallength - 1);
endfunction

% finds the peak of the parabola fitting the three points: f(x) = ax^2 + bx + c
% (a,b,c) determined by f([-1 0 1]) = vec
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

% make a plot of a series of HRTFs in the time domain
% x-axis: time
% y-axis: azimuth angle of HRTF
% z-axis: amplitude
function imshow_interpolation(l_eq_hrir_S, r_eq_hrir_S, first, last, steps)
	% samples to show from the impulse response
	ir_length = 128;
	% upsample the impulse responses by this factor
	upsampling = 4;

	image_left  = [];
	image_right = [];

	for index=linspace(first, last, steps)
		impulse_resps = interpolate_impulse_response_simple(l_eq_hrir_S, r_eq_hrir_S, index);
		ir_left = resample(impulse_resps(1:ir_length, 1), upsampling, 1);
		ir_right = resample(impulse_resps(1:ir_length, 2), upsampling, 1);
		image_left = [image_left; ir_left'];
		image_right = [image_right; ir_right'];
	endfor

	subplot(121)
	axis('tight', [0 ir_length 0 360]);
	imagesc(image_left, [-1 1]);
	subplot(122)
	axis('tight', [0 ir_length 0 360]);
	imagesc(image_right, [-1 1]);
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

filename = 'binauraltest.wav';
% audiowrite(filename, circle(input_signal, 0, 15, l_eq_hrir_S, r_eq_hrir_S), 44100);
audiowrite(filename, continuous_circle_with_interpolation(rand([44100 * 2, 1])-0.5, 73, 96, 2000, l_eq_hrir_S, r_eq_hrir_S), 44100);
% disp(strcat('wrote:', filename));
