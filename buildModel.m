function [model, mu, range] = buildModel(dirName)
%BUILDMODEL Builds a RBM
%   Detailed explanation goes here

files = getAllFiles(dirName);
window = windows('hanning');
X = zeros(numel(files), 4000);

counter = 0;
for i = 1:numel(files)
    filename = char(files(i));
    rev = fliplr(filename);
    label = rev(12);
    if strcmp(label, 'O')
        continue;
    end
        
    fileName = char(files(i));
    [s, fs] = readwav(fileName);
    frames = enframe(s, window, length(window) / 2)';
    F = rfft(frames);
    f = F(:)';
    
    counter = counter + 1;
    if (length(f) > 4000)
        X(counter,:) = f(1:4000);
    else
        X(counter,1:length(f)) = f;
    end
end

X = X(1:counter,:);

%% Feature scaling

mu = mean(X);
range = max(X) - min(X);

% Scale features
X = (X - repmat(mu, numel(files), 1)) ./ repmat(range, numel(files), 1);
% To [0,1] scale
X = (X + 1) / 2;

%% RBM
model = rbmBB(X, 1000, 'maxepoch', 10, 'verbose', true);

end

