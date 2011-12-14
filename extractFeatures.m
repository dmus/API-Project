function [f] = extractFeatures(model, mu, range, filename)
%EXTRACTFEATURES Summary of this function goes here
%   Detailed explanation goes here
[s, fs] = readwav(filename);
window = windows('hanning');
frames = enframe(s, window, length(window) / 2)';
F = rfft(frames);
v = zeros(size(mu));

Fprime = F(:)';
v(1:length(Fprime)) = Fprime;
v = v(1:4000);
% Scale features

v = (v - mu) ./ range;

% To [0,1] scale
v = (v + 1) / 2;

f = rbmVtoH(model, v);
end
