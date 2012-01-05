[model, mu, range] = buildModel('data/training');

files = getAllFiles('data/training');

%% Build training and test matrices
trainingSet = zeros(numel(files), 100);
trainingLabels = zeros(numel(files), 1);

counter = 0;
for i = 1:numel(files)
    filename = char(files(i));
    rev = fliplr(filename);
    label = rev(12);
    if strcmp(label, 'O')
        continue;
    elseif strcmp(label, 'Z')
        label = '0';
    end
    counter = counter + 1;
    trainingLabels(counter) = str2double(label);
    
    trainingSet(counter,:) = extractFeatures(model, mu, range, filename);
end

trainingSet = trainingSet(1:counter,:);
trainingLabels = trainingLabels(1:counter);

files = getAllFiles('data/test');

testSet = zeros(numel(files), 100);
testLabels = zeros(numel(files), 1);

counter = 0;
for i = 1:numel(files)
    filename = char(files(i));
    rev = fliplr(filename);
    label = rev(12);
    if strcmp(label, 'O')
        continue;
    elseif strcmp(label, 'Z')
        label = '0';
    end
    
    counter = counter + 1;
    testLabels(counter) = str2double(label);
    
    %testSet(counter,:) = extractMfccFeatures(filename);
    testSet(counter,:) = extractFeatures(model, mu, range, filename);
end

testSet = testSet(1:counter,:);
testLabels = testLabels(1:counter);

%% Classifying with SVM
svm = svmtrain(trainingLabels, trainingSet);
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, svm);