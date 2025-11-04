clc; clear; close all;

%% 1. Load images
imgFolder = fullfile('dataset','images');
classes = {'UAVs','Thunder','Birds','Planes'};
imds = imageDatastore(fullfile(imgFolder,classes), ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');

inputSize = [241 241];
imds.ReadFcn = @(filename)imresize(imread(filename),inputSize);

[imdsTrain, imdsTest] = splitEachLabel(imds,0.8,'randomized');
numTrainImg = numel(imdsTrain.Files);
numTestImg = numel(imdsTest.Files);

%% 2. Load RF data
rfTable = readtable('UAV_RF.csv'); 
rfData = table2array(rfTable(:,1:end-1));
rfLabels = categorical(rfTable.Label);

% Optional: Normalize RF signals
rfData = rfData./max(abs(rfData),[],2);

%% 3. Align RF data with image count
numRF = size(rfData,1);

% Reduce to minimum size to avoid index errors
numTrain = min(numTrainImg, floor(0.8*numRF));
numTest  = min(numTestImg, numRF - numTrain);

idxRF = randperm(numRF);
rfTrain = rfData(idxRF(1:numTrain),:);
rfTest  = rfData(idxRF(numTrain+1:numTrain+numTest),:);
rfTrainLabels = rfLabels(idxRF(1:numTrain));
rfTestLabels  = rfLabels(idxRF(numTrain+1:numTrain+numTest));

% Adjust image lists to match RF count
imdsTrain.Files = imdsTrain.Files(1:numTrain);
imdsTest.Files  = imdsTest.Files(1:numTest);

%% 4. Create combined dataset
trainData = table(imdsTrain.Files, rfTrain, rfTrainLabels, 'VariableNames',{'ImageFile','RF','Label'});
testData  = table(imdsTest.Files, rfTest, rfTestLabels, 'VariableNames',{'ImageFile','RF','Label'});

%%% 5. Display sample
%sampleImg = imresize(imread(trainData.ImageFile{1}), inputSize);
%sampleRF  = trainData.RF(1,:);

%figure;
%subplot(1,2,1); imshow(sampleImg); title('Sample UAV Image');
%subplot(1,2,2); plot(sampleRF); title('Sample RF Signal');

%% 6. Save dataset
save('QCNN_dataset.mat','trainData','testData','inputSize');
fprintf('Saved combined dataset to QCNN_dataset.mat\n');
