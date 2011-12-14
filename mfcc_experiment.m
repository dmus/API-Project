[trainingSetMan, trainingLabelsMan] = getMfccDatasetFromDir('data/training/MAN/');
[testSetMan, testLabelsMan] = getMfccDatasetFromDir('data/test/MAN/');

[trainingSetWoman, trainingLabelsWoman] = getMfccDatasetFromDir('data/training/WOMAN/');
[testSetWoman, testLabelsWoman] = getMfccDatasetFromDir('data/test/WOMAN/');

trainingSet = [trainingSetMan; trainingSetWoman];
trainingLabels = [trainingLabelsMan; trainingLabelsWoman];

testSet = [testSetMan; testSetWoman];
testLabels = [testLabelsMan; testLabelsWoman];

model = svmtrain(trainingLabels, trainingSet);
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, model);