function label = getLabelByFilename(filename)
%GETLABELBYFILENAME Summary of this function goes here
%   Detailed explanation goes here
    rev = fliplr(filename);
    label = rev(12);
    if strcmp(label, 'Z')
        label = '0';
    end
    
    label = str2double(label);
end

