function [X, mu, minVal, maxVal] = scaleSet(X, mu, minVal, maxVal)
%SCALESET Summary of this function goes here
%   Detailed explanation goes here

if nargin == 1
    mu = mean(X);
    maxVal = max(X);
    minVal = min(X);
end

range = maxVal - minVal;
signedRangeInverse = 1 ./ range;

% Scale features
X = bsxfun(@times, bsxfun(@minus, X, minVal), signedRangeInverse) * 2 - 1;
X = (X + 1) / 2;
end

