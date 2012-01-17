% simple wrapper for sigmoid function
function [y] = sigmoid(x)

    y = 1./(1 + exp (-x));

