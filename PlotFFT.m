function [ x ] = PlotFFT( filename )
%PlotFFT Plots the FFT of a wav file

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

fprintf('Calculate the FFT of the signal y: z = fft(y)\n');
z = fft(y);

fprintf('Shift the FFT z to get a symmetric envelope: z = fftshift(z)\n');
z = fftshift(z);

fprintf('Calculate the power = abs value of the FFT: z = abs(z)\n');
z = abs(z);

fprintf('And plot it: plot(z)\n');
plot(z);

fprintf('Use ylim([0 10000000]); to rescale the y-axis.\n');

fprintf('\nUse [x] = PlotFFT() to get the FFT of the wav in the variable x.\n');
x = y;