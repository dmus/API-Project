files = getAllFiles('data/training');

window = windows('hanning');

%% Build training and test matrices
trainingSet = zeros(numel(files), 258);
trainingLabels = zeros(numel(files), 1);

for i = 1:numel(files)
    filename = char(files(i));
    label = getLabelByFilename(filename);
 
    trainingLabels(i) = label;
    
    [y, fs] = readwav(filename);
    frames = enframe(y, window, length(window) / 2)';
    F = rfft(frames);
    f = [mean(F') std(F')];
    trainingSet(i,:) = f;
end

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
    f = [mean(F') std(F')];
    testSet(i,:) = f;
end

%% Classifying with SVM
svm = svmtrain(trainingLabels, trainingSet);
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, svm);