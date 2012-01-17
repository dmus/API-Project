addpath('Voicebox');

%[model, mu, minVal, maxVal] = buildModel('data/training/man');

files = getAllFiles('data/training');

%% Build training and test matrices
n = 6;
window = windows('hanning');
X = zeros(0, n * 129);

trainingLabels = zeros(0,1);
for i = 1:numel(files)
    filename = char(files(i));    
    label = getLabelByFilename(filename);
    [y, fs] = readwav(filename);
    frames = enframe(y, window, length(window) / 2)';
    F = rfft(frames)';
    
    m = size(F, 1);
    A = zeros(m - n + 1, n * 129);
    
    start = 1;
    stop = n;
    while stop <= size(F, 1)
        T = F(start:stop,:);
        A(start,:) = T(:);
        
        start = start + 1;
        stop = stop + 1;
    end
    
    trainingLabels(size(X,1) + 1:size(X,1) + size(A,1), :) = label;
    X = [X; A];
end

X = abs(X);
X = log10(X);
%X = X .* conj(X);
%X = bsxfun(@minus,X,mean(X));
%X = bsxfun(@rdivide, X, std(X));
%X = scaleSet(X, mu, minVal, maxVal);

%% RBM
% shuffle the training data
perm_idx = randperm (size(X,1));
X = X(perm_idx, :);
trainingLabels = trainingLabels(perm_idx);

% construct RBM and use default configurations
%R = default_rbm (size(X, 2), 300);

% use continuous values
%R.data.binary = 0;

% set grbm parameters
%R.grbm.do_vsample = 1;
%R.grbm.do_normalize = 1;
%R.grbm.do_normalize_std = 1;
%R.grbm.learn_sigmas = 1;

% max. 100 epochs
%R.iteration.n_epochs = 100;

% set the stopping criterion
%R.stop.criterion = 1;
%R.stop.recon_error.tolerate_count = 1000;

% save the intermediate data after every epoch
%R.hook.per_epoch = {@save_intermediate, {'grbm_faces.mat'}};

% print learining process
%R.verbose = 1;

% display the progress
%R.debug.do_display = 0;
%R.debug.display_interval = 10;
%R.debug.display_fid = 1;
%R.debug.display_function = @visualize_grbm;

% train RBM
%fprintf(1, 'Training GB-RBM\n');
%tic;
%R = train_rbm (R, X);
%fprintf(1, 'Training is done after %f seconds\n', toc);
%%

%X = real(X);
%X = scaleSet(X, mu, minVal, maxVal);

trainingSet = zeros(size(X,1), 300);
for i = 1:size(X,1)
    trainingSet(i,:) = rbm_get_hidden(X(i,:), R);
end

%% Test set
files = getAllFiles('data/test');
X = zeros(0, n * 129);
testLabels = zeros(0,1);
w = zeros(numel(files), 1);
for i = 1:numel(files)
    filename = char(files(i));    
    label = getLabelByFilename(filename);
    [y, fs] = readwav(filename);
    frames = enframe(y, window, length(window) / 2)';
    F = rfft(frames)';
    
    m = size(F, 1);
    A = zeros(m - n + 1, n * 129);
    
    start = 1;
    stop = n;
    while stop <= size(F, 1)
        T = F(start:stop,:);
        A(start,:) = T(:);
        
        start = start + 1;
        stop = stop + 1;
    end
    w(i) = size(A,1);
    testLabels(size(X,1) + 1:size(X,1) + size(A,1), :) = label;
    X = [X; A];
end
X = abs(X);
X = log10(X);
%X = X .* conj(X);



testSet = zeros(size(X,1), 300);
for i = 1:size(X,1)
    testSet(i,:) = rbm_get_hidden(X(i,:), R);
end

%% Classifying with SVM
svm = svmtrain(trainingLabels, trainingSet);
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, svm);

%% Majority vote
offset = 1;
predictions = zeros(size(w));
correct = zeros(size(w));
for i = 1:length(w)
    len = w(i);
    correct(i) = testLabels(offset);
    predicted = predictedLabels(offset:offset + len - 1);
    predicted(predicted == 0) = 10;
    
    % Majority vote
    votes = zeros(10,1);
    
    for j = 1:length(predicted)
        votes(predicted(j)) = votes(predicted(j)) + 1;
    end
    
    [~, index] = max(votes);
    vote = index;
    if index == 10
        vote = 0;
    end
    
    predictions(i) = vote;
    
    offset = offset + len;
end