function playFile(filename)
%PLAYFILE Summary of this function goes here
%   Detailed explanation goes here

    [s, fs] = readwav(filename);
    sound(s,fs);
end

