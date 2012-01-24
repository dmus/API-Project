function [X, y, w] = getMfccDatasetFromDir(dirName, sumStats)
%GETMFCCDATASETFROMDIR Puts features for every example in dir in matrix X, labels
%in y

files = getAllFiles(dirName);

if sumStats
    X = zeros(0,84);
else
    X = zeros(0,42);
end

y = zeros(0,1);
w = zeros(0,1);

for i = 1:numel(files)
    % Extract label
    filename = char(files(i));
    label = getLabelByFilename(filename);
    
    % MFCCs
    C = extractMfccFeatures(filename);
    
    if sumStats
        C = [mean(C) std(C)];
    else
        windowSize = 1;
        D = zeros(0,42);
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

