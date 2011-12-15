function [f] = extractMfccFeatures(filename)
%EXTRACTFEATURES Extract MFFC features, aggregrate and return them in row vector
%   Features extracted include log energy, 0th cepstral coef, delta and
%   delta-delta coefs
[y, fs] = readwav(filename);

C = melcepst(y, fs, 'NE0dD',12,floor(3*log(fs)),256); % Hanning window, include log energy, 0th cepstral coef, delta and delta-delta coefs

f = [mean(C) std(C) min(C) max(C)];

end

