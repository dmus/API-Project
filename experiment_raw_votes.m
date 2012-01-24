% Experiment with power spectrum features and voting classification

addpath('Voicebox');

fprintf('Building trainingset...\n');
[trainingSet, trainingLabels] = getDatasetFromDir('data/training',0);

fprintf('Building testset...\n');
[testSet, testLabels, w] = getDatasetFromDir('data/test',0);

fprintf('Training...\n');
model = svmtrain(trainingLabels, trainingSet, '-b 1');

fprintf('Testing...\n');
[predictedLabels, accuracy, dec_values] = svmpredict(testLabels, testSet, model, '-b 1');

fprintf('Counting votes...\n');
predictions = count_votes(dec_values,w);
correct = testLabels(cumsum(w));

performance = sum(predictions==correct)/length(predictions);
fprintf('Performance: %2.4f (%i/%i)\n', performance, sum(predictions==correct), length(predictions));
