%[model, mu, range] = buildModel('data/training/man');

files = getAllFiles('data/training/MAN');

%% Build training and test matrices
n = 6;
window = windows('hanning');
X = zeros(0, n * 129);

trainingLabels = zeros(0,1);
for i = 1:numel(files)
    filename = char(files(i));    
    label = getLabelByFilename(filename);
    [y, fs] = readwav(filename);
    frames = enframe(y, window, length(window) / 2)';
    F = rfft(frames)';
    
    m = size(F, 1);
    A = zeros(m - n + 1, n * 129);
    
    start = 1;
    stop = n;
    while stop <= size(F, 1)
        T = F(start:stop,:);
        A(start,:) = T(:);
        
        start = start + 1;
        stop = stop + 1;
    end
    
    trainingLabels(size(X,1) + 1:size(X,1) + size(A,1), :) = label;
    X = [X; A];
end
X = abs(X);
X = scaleSet(X);

trainingSet = zeros(size(X,1), 200);
for i = 1:size(X,1)
    trainingSet(i,:) = rbmVtoH(model, X(i,:));
end

%% Test set
files = getAllFiles('data/test/MAN');
X = zeros(0, n * 129);
testLabels = zeros(0,1);
w = zeros(numel(files), 1);
for i = 1:numel(files)
    filename = char(files(i));    
    label = getLabelByFilename(filename);
    [y, fs] = readwav(filename);
    frames = enframe(y, window, length(window) / 2)';
    F = rfft(frames)';
    
    m = size(F, 1);
    A = zeros(m - n + 1, n * 129);
    
    start = 1;
    stop = n;
    while stop <= size(F, 1)
        T = F(start:stop,:);
        A(start,:) = T(:);
        
        start = start + 1;
        stop = stop + 1;
    end
    w(i) = size(A,1);
    testLabels(size(X,1) + 1:size(X,1) + size(A,1), :) = label;
    X = [X; A];
end
X = abs(X);
X = scaleSet(X);

testSet = zeros(size(X,1), 200);
for i = 1:size(X,1)
    testSet(i,:) = rbmVtoH(model, X(i,:));
end

%% Classifying with SVM
%svm = svmtrain(trainingLabels, trainingSet);
[predictedLabels, accuracy] = svmpredict(testLabels, testSet, svm);

%% Majority vote
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