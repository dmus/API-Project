% Experiment with rbm features and voting classification

addpath('Voicebox');
addpath('rbm');

num_features = 300;

fprintf('Building trainingset...\n');
[trainingSet, trainingLabels] = getDatasetFromDir('data/training',0);

fprintf('Building and training Restricted Boltzmann Machine...\n');
rbm = build_rbm(trainingSet, num_features);

fprintf('Building testset...\n');
[testSet, testLabels, w] = getDatasetFromDir('data/test',0);

fprintf('Extracting features...\n');
training = zeros(size(trainingSet,1), num_features);
for i = 1:size(trainingSet,1)
    training(i,:) = rbm_get_hidden(trainingSet(i,:), rbm); 
end

testing = zeros(size(testSet,1), num_features);
for i = 1:size(testSet,1)
    testing(i,:) = rbm_get_hidden(testSet(i,:), rbm); 
end

fprintf('Training...\n');

model = svmtrain(trainingLabels, training, '-b 1 -h 0');

fprintf('Testing...\n');
[predictedLabels, accuracy, dec_values] = svmpredict(testLabels, testing, model, '-b 1');

fprintf('Counting votes...\n');
predictions = count_votes(dec_values,w);
correct = testLabels(cumsum(w));

performance = sum(predictions==correct)/length(predictions);
fprintf('Performance: %2.4f (%i/%i)\n', performance, sum(predictions==correct), length(predictions));
