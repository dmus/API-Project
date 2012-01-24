function [X, y, w] = getDatasetFromDir(dirName, sumStats)
%GETMFCCDATASETFROMDIR Puts features for every example in dir in matrix X, labels
%in y
window = windows('hanning');
files = getAllFiles(dirName);

if sumStats
    X = zeros(0,258);
else
    X = zeros(0,129);
end

y = zeros(0,1);
w = zeros(0,1);

for i = 1:numel(files)
    % Extract label
    filename = char(files(i));
    label = getLabelByFilename(filename);
    
    % MFCCs
    [s, fs] = readwav(filename);
    frames = enframe(s, window, length(window) / 2)';
    C = rfft(frames);
    C = log10(abs(C)');
    
    if sumStats
        C = [mean(C) std(C)];
    else
        windowSize = 1;
        D = zeros(0,129);
        for j = 1 : size(C,1)
            if mod(j, windowSize) == 0
                T = C(j - windowSize + 1:j,:);
                D = [D; T];
                %D = [D; mean(T,1) std(T,1)];
            end
        end
        C = D;
    end
    
    X = [X; C];
    
    
    w = [w; size(C,1)];
    labels = zeros(size(C,1),1);
    labels(:) = label;
    
    y = [y; labels];
end

end

