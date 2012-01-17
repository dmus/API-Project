function [ x ] = PlotWav( filename )
%PlotFFT Plots the wav file

%Determine how many arguments are passed to the function
if (nargin == 0)
    %If no arguments are given, we give it a default value
    filename = 'sound.wav';
end

fprintf('We read a wav-file: %s\n',filename); 
[y,fs,mode,fidx] = readwav(filename,'r',-1,0);
fprintf('And plot it in a graph.\n');
fprintf('Note it is a signed 16 bits sound file.\n'); 
plot(y);
fprintf('\nUse [x] = PlotWav() to get the wav in the variable x.\n');
x = y;