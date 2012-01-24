addpath('Voicebox');



files = getAllFiles('data/training');

window = windows('hanning');

%% Build training and test matrices
fprintf('Building trainingset...\n');
trainingSet = zeros(numel(files), 258);
trainingLabels = zeros(numel(files), 1);

for i = 1:numel(files)
    filename = char(files(i));
    label = getLabelByFilename(filename);
 
    trainingLabels(i) = label;
    
    [y, fs] = readwav(filename);
    frames = enframe(y, window, length(window) / 2)';
    F = rfft(frames);
    F = log10(abs(F));
    f = [mean(F') std(F')];
    trainingSet(i,:) = f;
end

fprintf('Building testset...\n');
files = getAllFiles('data/test');

testSet = zeros(numel(files), 258);
testLabels = zeros(numel(files), 1);

for i = 1:numel(files)
    filename = char(files(i));
    label = getLabelByFilename(filename);
    
    testLabels(i) = label;
    
    [y, fs] = readwav(filename);
    frames = enframe(y, window, length(window) / 2)';
    F = rfft(frames);
    F = log10(abs(F));
    f = [mean(F') std(F')];
    testSet(i,:) = f;
end

%% Classifying with SVM
fprintf('Training...\n');
svm = svmtrain(trainingLabels, trainingSet);

fprintf('Testing...\n');
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, svm);

performance = sum(predictedLabels==testLabels)/length(testLabels);
fprintf('Performance: %2.4f (%i/%i)\n', performance, sum(predictedLabels==testLabels), length(testLabels));