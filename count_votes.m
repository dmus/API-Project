function predictions = count_votes(dec_values, len)
%VOTE Summary of this function goes here
%   Detailed explanation goes here
    
    predictions = zeros(size(len));
    cum = cumsum([0; len]);    
    for i = 1:length(len)
        [m,index] = max(mean(dec_values(cum(i) + 1:cum(i+1),:)));
        predictions(i) = index;
    end

    predictions(predictions == 10) = 0;
end

