function [f] = preprocess(filename)
%PREPROCESS Summary of this function goes here
%   Detailed explanation goes here
    window = windows('hanning');
    [s, fs] = readwav(filename);
    frames = enframe(s, window, length(window) / 2)';
    F = rfft(frames);
    
    F = F(1:100, :);
    if size(F,2) > 40
        F = F(:,1:40);
    else
        Temp = F;
        F = zeros(100, 40);
        F(:,1:size(Temp,2)) = Temp;
    end
    
    G = zeros(100,10);
    column = 0;
    for j = 1:size(F,2)
        if mod(j,4) == 0
            column = column + 1;
            G(:,column) = mean(F(:,j-3:j),2)';
        end
    end
    
    f = G(:)';

end

