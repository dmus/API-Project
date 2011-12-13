[model, mu, range] = buildModel('data/training');

files = getAllFiles('data/training');

%% Build training and test matrices
trainingSet = zeros(numel(files), 500);
trainingLabels = zeros(numel(files), 1);


for i = 1:numel(files)
    filename = char(files(i));
    rev = fliplr(filename);
    label = rev(12);
    if strcmp(label, 'O')
        continue
    elseif strcmp(label, 'Z')
        label = 0;
    end
    
    trainingLabels(i) = label;
    trainingSet(i,:) = extractFeatures(model, mu, range, filename);
end

files = getAllFiles('data/test');

testSet = zeros(numel(files), 500);
testLabels = zeros(numel(files), 1);

for i = 1:numel(files)
    filename = char(files(i));
    rev = fliplr(filename);
    label = rev(12);
    if strcmp(label, 'O')
        continue
    elseif strcmp(label, 'Z')
        label = 0;
    end
    
    testLabels(i) = label;
    testSet(i,:) = extractFeatures(model, mu, range, filename);
end

%% Classifying with SVM
svm = svmtrain(trainingLabels, trainingSet);
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, svm);