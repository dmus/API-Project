[s, fs] = readwav('data/test/WOMAN/JW/4A_endpt.wav');
[s2, fs2] = readwav('data/test/WOMAN/JW/3A_endpt.wav');
s2(2641:4080) = 0;
%spgrambw(s,fs,'pJcw');

window = windows('hanning');
frames = enframe(s, window, length(window) / 2)';
F = rfft(frames);

frames2 = enframe(s2, window, length(window) / 2)';
F2 = rfft(frames2);
%% RBM part
data(1,:) = F(:)';
data(2,:) = F2(:)';

mu = mean(data);
range = max(data) - min(data);

% Scale features
data = (data - repmat(mu, 2, 1)) ./ repmat(range, 2, 1);
% To [0,1] scale
data = (data + 1) /2;

model = rbmBB(data, 100, 'maxepoch', 100, 'verbose', true);

h = rbmVtoH(model, data(1,:));
v = rbmHtoV(model, h);

%% Reconstruction
v = v * 2 - 1;
v = v .* range + mu;

R = reshape(v, size(F));

Fprime = irfft(R);
reconstructed = overlapadd(Fprime', window, length(window) / 2);

sound(s, fs);
sound(reconstructed, fs);