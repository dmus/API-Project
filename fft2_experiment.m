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

trainingSet = abs(trainingSet);
%% Scale
[trainingSet, mu, minVal, maxVal] = scaleSet(trainingSet);
%trainingSet = (trainingSet - repmat(mu, size(trainingSet, 1), 1));% ./ repmat(range, size(trainingSet, 1), 1);
% To [0,1] scale
%trainingSet = (trainingSet + 1) / 2;

%% Test set

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

%% Extra

% Scale features
testSet = abs(testSet);
testSet = scaleSet(testSet, mu, minVal, maxVal);
%testSet = (testSet - repmat(mu, size(testSet, 1), 1));% ./ repmat(range, size(testSet, 1), 1);
% To [0,1] scale
%testSet = (testSet + 1) / 2;

%% Classifying with SVM
svm = svmtrain(trainingLabels, trainingSet);
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, svm);