clc
clear all
close all

n=1;

a=ones(n,1);
t=zeros(n,1);

%%
for i=1:n
    tic
%% Load image from Current Working Folder

%Build full file name from parts.
imageFolder = fullfile('round_set1_augmented');

%Create an ImageDatastore to work with collections of images

%IMDS = imageDatastore(__,'IncludeSubfolders',TF) specifies the 
%logical true to indicate to included recursively, the folder and its subfolders

%IMDS = imageDatastore(__,'LabelSource',SOURCE)
%If SOURCE is 'foldernames', then the values for the Labels property are obtained from the folder names of the image files.
imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

%Count the number of times each unique label occurs
tbl = countEachLabel(imds);     %Table made of pictures

% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2}); 
maxNumImages = 1169;
minSetCount = min(maxNumImages,minSetCount);

% Use splitEachLabel method to trim the bigger set, to make both equal
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds);

%% Load pretrained network
net = resnet50();
% alexnet, googlenet, vgg16, and vgg19 sind andere bekannte

% Visualize the first section of the network. 
% figure
% plot(net)
% title('First section of ResNet-50')
% set(gca,'YLim',[150 170]);

% % Inspect the first layer
% net.Layers(1)
% % Inspect the last layer
% net.Layers(end)
% % Number of class names for ImageNet classification task
% numel(net.Layers(end).ClassNames)

%% prepare Training and Test Image Sets

trainingSet = imds;

%imageFolder2 = fullfile('testSet');

%imds2 = imageDatastore(imageFolder2, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

%testSet = imds2;
%[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');

% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
%augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');
% Generate batches of augmented image data

%% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);      % Convert matrix to intensity image.
w1 = imresize(w1,5);    % scales the image by 5 times

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
% figure
% montage(w1)
% title('First convolutional layer weights')
featureLayer = 'fc1000';

%Y = activations(net, X, layerOut) returns network activations for a specific layer using the network net and the data X
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

%% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns')

%% Extract test features using the CNN
%testFeatures = activations(net, augmentedTestSet, featureLayer, ...
%    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Pass CNN image features to trained classifier
%predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns')


% Get the known labels
%testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
%confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
%confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% Apply trained Classifier on one Test Image
%testImage = readimage(testSet,1);
%testLabel = testSet.Labels(1);
% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
%ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
%imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

% make a prediction using the classifier
%predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');

% How good is the predicttion?
%accuracy(i,1)=mean(diag(confMat));
%cMat(i,1)={confMat};
 
t(i,1)=toc;
end

save Traineddata;

%%
% make a prediction using the classifier
%predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')