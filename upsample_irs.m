load recherche.ircam.fr/COMPENSATED/MAT/HRIR/IRC_1032_C_HRIR.mat
% load recherche.ircam.fr/RAW/MAT/HRIR/IRC_1026_R_HRIR.mat

% makes further calculations faster by upsampling the impulse responses in advance
% and precalculating the delays

% return a struct A such that:
% A.diffs_left(i,j) = delaydifference(left(i), left(j))
% A.diffs_right(i,j) = delaydifference(right(i), right(j))
% A.irs_{left,right}(i,:) =  HRTF impulse response at index i, upsampled <A.upsampling> times
% Note that the delay differences refer to non-upsampled samples
function irs_and_delaydiffs = upsample_irs(l_eq_hrir_S, r_eq_hrir_S, upsampling)
	diffs_left = zeros(187);
	diffs_right = zeros(187);

	% calculate the delay difference matrices
	% {{{
	% fill the left upper triangle of the matrices
	for i=(1:187)
		for j=(i+1:187)
			disp(strcat('i=', num2str(i), ',j=', num2str(j))); fflush(stdout);
			diffs_left(i, j) = delaydifference(l_eq_hrir_S.content_m(i,:), l_eq_hrir_S.content_m(j,:), upsampling);
			diffs_right(i, j) = delaydifference(r_eq_hrir_S.content_m(i,:), r_eq_hrir_S.content_m(j,:), upsampling);
		endfor
	endfor

	% use antisymmetry of delaydifference to fill the rest of the matrices
	diffs_left  = diffs_left  - diffs_left';
	diffs_right = diffs_right - diffs_right';
	% }}}

	% calculate the upsampled signals
	% {{{
	irs_left = zeros(187, 512 * upsampling);
	irs_right = zeros(187, 512 * upsampling);

	for i=(1:187)
		disp(num2str(i));
		irs_left(i,:) = resample(l_eq_hrir_S.content_m(i,:), upsampling, 1);
		irs_right(i,:) = resample(r_eq_hrir_S.content_m(i,:), upsampling, 1);
	endfor

	irs_and_delaydiffs = struct(
		'upsampling', upsampling,
		'diffs_left', diffs_left, 'diffs_right', diffs_right,
		'irs_left', irs_left, 'irs_right', irs_right
	);
	% }}}

	save -6 'irs_and_delaydiffs.mat' irs_and_delaydiffs
endfunction

% Finds the delay difference between two HRTFs a and b
% For decent results choose two HRTFs that are 'close' to each other
% returns the delay between the responses (delay > 0 if b comes after a)
function diff = delaydifference(signal_a, signal_b, upsampling)
	% business logic
	assert (length(signal_a) == length(signal_b))
	signallength = length(signal_a);

	% cross correlate the signal and upsample
	% cross correlation = convolution but without the time reversal that makes convolution commutative
	crosscorr_upsampled = resample(fftconv(fliplr(flipud(signal_a)), signal_b), upsampling, 1);

	% find the peak
	[value, peak_index] = max(crosscorr_upsampled);
	peak_index += parabolic_interpolation(crosscorr_upsampled(peak_index-1:peak_index+1)) - 1;

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
