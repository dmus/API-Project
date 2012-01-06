function [model, mu, minVal, maxVal] = buildModel(dirName)
%BUILDMODEL Builds a RBM
%   Detailed explanation goes here

n = 6;
window = windows('hanning');
files = getAllFiles(dirName);
X = zeros(0, n * 129);

for i = 1:numel(files)
    filename = char(files(i));    
    [y, fs] = readwav(filename);
    frames = enframe(y, window, length(window) / 2)';
    F = rfft(frames)';
    
    m = size(F, 1);
    A = zeros(m - n + 1, n * 129);
    
    start = 1;
    stop = n;
    while stop <= size(F, 1)
        T = F(start:stop,:);
        A(start,:) = T(:);
        
        start = start + 1;
        stop = stop + 1;
    end
    
    X = [X; A];
end
X = abs(X);
[X, mu, minVal, maxVal] = scaleSet(X);
%% RBM
model = rbmBB(X, 200, 'maxepoch', 10, 'verbose', true);

end

