function [f] = extractFeatures(model, mu, range, filename)
%EXTRACTFEATURES Summary of this function goes here
%   Detailed explanation goes here

f = preprocess(filename);
    
% Scale features

f = (f - mu) ./ range;

% To [0,1] scale
f = (f + 1) / 2;

f = rbmVtoH(model, f);
end
