function C = extractMfccFeatures(filename)
%EXTRACTFEATURES Extract MFFC features, return them in matrix, each frame
%occupies one row
%   Features extracted include log energy, 0th cepstral coef, delta and
%   delta-delta coefs
[y, fs] = readwav(filename);

C = melcepst(y, fs, 'NE0dD',12,floor(3*log(fs)),256); % Hanning window, include log energy, 0th cepstral coef, delta and delta-delta coefs

end

