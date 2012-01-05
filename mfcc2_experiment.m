% Experiment for MFCC features

[trainingSet, trainingLabels] = getMfccDatasetFromDir('data/training');
[testSet, testLabels, w] = getMfccDatasetFromDir('data/test');

model = svmtrain(trainingLabels, trainingSet);
%load('svm_frames.mat');
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, model);

offset = 1;
predictions = zeros(size(w));
correct = zeros(size(w));
for i = 1:length(w)
    len = w(i);
    correct(i) = testLabels(offset);
    predicted = predictedLabels(offset:offset + len - 1);
    predicted(predicted == 0) = 10;
    
    % Majority vote
    votes = zeros(10,1);
    
    for j = 1:length(predicted)
        votes(predicted(j)) = votes(predicted(j)) + 1;
    end
    
    [~, index] = max(votes);
    vote = index;
    if index == 10
        vote = 0;
    end
    
    predictions(i) = vote;
    
    offset = offset + len;
end