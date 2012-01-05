% Experiment for MFCC features

[trainingSet, trainingLabels] = getMfccDatasetFromDir('data/training');
[testSet, testLabels] = getMfccDatasetFromDir('data/test');

model = svmtrain(trainingLabels, trainingSet);
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, model);