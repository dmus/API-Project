function [model, mu, range] = buildModel(dirName)
%BUILDMODEL Summary of this function goes here
%   Detailed explanation goes here

files = getAllFiles(dirName);
window = windows('hanning');
X = zeros(numel(files), 4500);

for i = 1:numel(files)
    fileName = char(files(i));
    [s, fs] = readwav(fileName);
    frames = enframe(s, window, length(window) / 2)';
    F = rfft(frames);
    f = F(:)';
    X(i,1:length(f)) = f;
end

%% Feature scaling

mu = mean(X);
range = max(X) - min(X);

% Scale features
X = (X - repmat(mu, numel(files), 1)) ./ repmat(range, numel(files), 1);
% To [0,1] scale
X = (X + 1) /2;

%% RBM
model = rbmBB(X, 500, 'maxepoch', 30, 'verbose', true);

end

