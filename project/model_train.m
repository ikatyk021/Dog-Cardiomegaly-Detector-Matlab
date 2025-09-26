% 準備訓練數據
% X_train和y_train應該是包含多張影像的特徵向量和對應的標籤
mdl = fitcsvm(X_train, y_train); % 使用SVM模型


%%%%%
%clear all;
close all;
clc;
digitDatasetPath=fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds=imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labelCount = countEachLabel(imds);
numTrainFiles = 750;
[imdsTrain,imdsValidation]=splitEachLabel(imds,numTrainFiles,'randomize');
% layers = [
%     imageInputLayer([28 28 1])
%     convolution2dLayer(3,8,'Padding',1)
%     batchNormalizationLayer()
%     reluLayer()
%     maxPooling2dLayer(2,'Stride',2)
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer()
%     reluLayer()
%     fullyConnectedLayer(10)
%     softmaxLayer()
%     classificationLayer];
layers = layers_2;
options=trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net=trainNetwork(imdsTrain,layers,options);
