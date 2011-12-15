function [X, y] = getMfccDatasetFromDir(dirName)
%GETMFCCDATASETFROMDIR Puts features for every example in dir matrix X, labels
%in y
%   Detailed explanation goes here

files = getAllFiles(dirName);
X = zeros(numel(files), 168);
y = zeros(numel(files), 1);

counter = 0;
for i = 1:numel(files)
    filename = char(files(i));
    rev = fliplr(filename);
    label = rev(12);
    if strcmp(label, 'O')
        continue;
    end
        
    fileName = char(files(i));
    f = extractMfccFeatures(fileName);
    
    counter = counter + 1;
    X(counter,:) = f;
    
    if strcmp(label, 'Z')
        label = '0';
    end
    y(counter) = str2double(label);
end

X = X(1:counter,:);
y = y(1:counter,:);

end

