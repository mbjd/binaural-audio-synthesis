% makes further calculations faster by upsampling the impulse responses in advance
% and precalculating the delays

% return a struct A such that:
% A.diffs_left(i,j) = delaydifference(left(i), left(j))
% A.diffs_right(i,j) = delaydifference(right(i), right(j))
% A.irs_{left,right}(i,:) =  HRTF impulse response at index i, upsampled <A.upsampling> times
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
		irs_left(i,:) = resample(l_eq_hrir_S.content_m(i,:), upsampling, 1);
		irs_right(i,:) = resample(r_eq_hrir_S.content_m(i,:), upsampling, 1);
	endfor

	irs_and_delaydiffs = struct(
		'upsampling', upsampling,
		'diffs_left', diffs_left, 'diffs_right', diffs_right,
		'irs_left', irs_left, 'irs_right', irs_right
	);
	% }}}

	save 'irs_and_delaydiffs.mat' irs_and_delaydiffs
endfunction
