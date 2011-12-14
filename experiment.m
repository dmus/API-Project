files = getAllFiles('data/test');

lengths = zeros(numel(files),1);

for i = 1:numel(files)
    filename = char(files(i));
    [s, fs] = readwav(filename);
    
    frames = enframe(s, window, length(window) / 2)';
    F = rfft(frames);
    
    lengths(i) = size(F,2);
end

mean(lengths)
std(lengths)
min(lengths)
max(lengths)