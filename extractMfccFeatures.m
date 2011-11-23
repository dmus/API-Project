function [f] = extractMfccFeatures(model, filename)
%EXTRACTFEATURES Summary of this function goes here
%   Detailed explanation goes here
[y, fs, ~, ~] = readwav(filename);

C = melcepst(y, fs, 'Ne0dD'); % Hanning window, include log energy, 0th cepstral coef, delta and delta-delta coefs

f = [mean(C) std(C) min(C) max(C)];

end

