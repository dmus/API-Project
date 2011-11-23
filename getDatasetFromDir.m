function [X, y] = getDatasetFromDir(dirName)
%GETDATASETFROMDIR Summary of this function goes here
%   Detailed explanation goes here

dirInfo = dir(dirName);
X = zeros((length(dirInfo) - 2) * 20, 156);
y = zeros((length(dirInfo) - 2) * 20, 1);

count = 0;
for i = 1:numel(dirInfo)
    if strcmp(dirInfo(i).name, '.') || strcmp(dirInfo(i).name, '..')
        continue;
    end
    
    speakerDir = strcat(dirName, dirInfo(i).name, '/');
    digitsInfo = dir(speakerDir);
    for j = 1:numel(digitsInfo)
        if strcmp(digitsInfo(j).name, '.') || strcmp(digitsInfo(j).name, '..')
            continue;
        end
        
        if strcmp(digitsInfo(j).name(1), 'O')
            continue;
        elseif strcmp(digitsInfo(j).name(1), 'Z')
            label = 0;
        else
            label = str2double(digitsInfo(j).name(1));
        end
        
        count = count + 1;
        y(count) = label;
        X(count,:) = extractMfccFeatures(strcat(speakerDir, digitsInfo(j).name));
    end
end

end

