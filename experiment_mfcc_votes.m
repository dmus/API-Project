% Experiment with MFCC features and voting classification

fprintf('Building trainingset...\n');
[trainingSet, trainingLabels] = getMfccDatasetFromDir('data/training');

fprintf('Building testset...\n');
[testSet, testLabels, w] = getMfccDatasetFromDir('data/test');

fprintf('Training...\n');
model = svmtrain(trainingLabels, trainingSet, '-b 1');

fprintf('Testing...\n');
[predictedLabels, accuracy, dec_values] = svmpredict(testLabels, testSet, model, '-b 1');

fprintf('Counting votes...\n');
predictions = count_votes(dec_values,w);
correct = testLabels(cumsum(w));

performance = sum(predictions==correct)/length(predictions);
fprintf('Performance: %2.4f (%i/%i)\n', performance, sum(predictions==correct), length(predictions));
