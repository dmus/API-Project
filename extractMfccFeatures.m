function [f] = extractMfccFeatures(filename)
%EXTRACTFEATURES Extract MFFC features and return them in row vector
%   Features extracted include log energy, 0th cepstral coef, delta and
%   delta-delta coefs
[y, fs] = readwav(filename);

C = melcepst(y, fs, 'Ne0dD'); % Hanning window, include log energy, 0th cepstral coef, delta and delta-delta coefs

f = [mean(C) std(C) min(C) max(C)];

end

