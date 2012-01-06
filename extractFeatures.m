function [f] = extractFeatures(model, mu, range, filename)
%EXTRACTFEATURES Summary of this function goes here
%   Detailed explanation goes here

f = preprocess(filename);
    

f = rbmVtoH(model, f);
end
