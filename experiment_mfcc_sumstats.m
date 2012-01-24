% Experiment with MFCC features and summary statistics classification

addpath('Voicebox');

fprintf('Building trainingset...\n');
[trainingSet, trainingLabels] = getMfccDatasetFromDir('data/training',1);

fprintf('Building testset...\n');
[testSet, testLabels, w] = getMfccDatasetFromDir('data/test',1);

fprintf('Training...\n');
model = svmtrain(trainingLabels, trainingSet);

fprintf('Testing...\n');
[predictedLabels, accuracy, dec_values] = svmpredict(testLabels, testSet, model);

performance = sum(predictedLabels==testLabels)/length(testLabels);
fprintf('Performance: %2.4f (%i/%i)\n', performance, sum(predictedLabels==testLabels), length(testLabels));
