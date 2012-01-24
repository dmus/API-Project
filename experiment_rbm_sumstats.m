% Experiment with rbm features and voting classification

addpath('Voicebox');
addpath('rbm');

num_features = 300;

fprintf('Building trainingset...\n');
[trainingSet, trainingLabels, w_training] = getDatasetFromDir('data/training',0);

fprintf('Building and training Restricted Boltzmann Machine...\n');
rbm = build_rbm(trainingSet, num_features);

fprintf('Building testset...\n');
[testSet, testLabels, w_test] = getDatasetFromDir('data/test',0);

fprintf('Extracting features...\n');
training = zeros(size(trainingSet,1), num_features);
for i = 1:size(trainingSet,1)
    training(i,:) = rbm_get_hidden(trainingSet(i,:), rbm); 
end

testing = zeros(size(testSet,1), num_features);
for i = 1:size(testSet,1)
    testing(i,:) = rbm_get_hidden(testSet(i,:), rbm); 
end

training_sumstats = zeros(length(w_training), num_features * 2);
cum = cumsum([0; w_training]);
training_labels = zeros(length(w_training), 1);
for i = 1:length(w_training)
    training_sumstats(i,:) = [mean(training(cum(i) + 1: cum(i + 1),:)) std(training(cum(i) + 1: cum(i + 1),:))];
    training_labels(i) = trainingLabels(cum(i + 1));
end

test_sumstats = zeros(length(w_test), num_features * 2);
cum = cumsum([0; w_test]);
test_labels = zeros(length(w_training), 1);
for i = 1:length(w_test)
    test_sumstats(i,:) = [mean(testing(cum(i) + 1: cum(i + 1),:)) std(testing(cum(i) + 1: cum(i + 1),:))];
    test_labels(i) = testLabels(cum(i + 1));
end

fprintf('Training...\n');

model = svmtrain(training_labels, training_sumstats);

fprintf('Testing...\n');
[predictedLabels, accuracy] = svmpredict(test_labels, test_sumstats, model);

performance = sum(predictedLabels==test_labels)/length(test_labels);
fprintf('Performance: %2.4f (%i/%i)\n', performance, sum(predictedLabels==test_labels), length(test_labels));
