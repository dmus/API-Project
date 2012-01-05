function [X, y, w] = getMfccDatasetFromDir(dirName)
%GETMFCCDATASETFROMDIR Puts features for every example in dir in matrix X, labels
%in y

files = getAllFiles(dirName);
%X = zeros(numel(files), 168);
%y = zeros(numel(files), 1);
X = zeros(0,84);
y = zeros(0,1);
w = zeros(0,1);

for i = 1:numel(files)
    % Extract label
    filename = char(files(i));
    rev = fliplr(filename);
    label = rev(12);
    if strcmp(label, 'Z')
        label = '0';
    end
    label = str2double(label);
    
    % MFCCs
    C = extractMfccFeatures(filename);
    
    % If summary statistics
    %C = [mean(C) std(C) min(C) max(C)];
    
    % If texture windows
    windowSize = 2;
    D = zeros(0,84);
    for j = 1 : size(C,1)
        if mod(j, windowSize) == 0
            T = C(j - windowSize + 1:j,:);
            D = [D; mean(T) std(T)];
        end
    end
    C = D;
    
    X = [X; C];
    
    
    w = [w; size(C,1)];
    labels = zeros(size(C,1),1);
    labels(:) = label;
    
    y = [y; labels];
end

end

