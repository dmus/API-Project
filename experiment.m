[trainingSetMan, trainingLabelsMan] = getDatasetFromDir('data/training/MAN/');
[testSetMan, testLabelsMan] = getDatasetFromDir('data/test/MAN/');

[trainingSetWoman, trainingLabelsWoman] = getDatasetFromDir('data/training/WOMAN/');
[testSetWoman, testLabelsWoman] = getDatasetFromDir('data/test/WOMAN/');

trainingSet = [trainingSetMan; trainingSetWoman];
trainingLabels = [trainingLabelsMan; trainingLabelsWoman];

testSet = [testSetMan; testSetWoman];
testLabels = [testLabelsMan; testLabelsWoman];

model = svmtrain(trainingLabels, trainingSet);
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, model);