%% To Clean Up
clc
clear

j=0;
% Loading trained data
load Traineddata;

% Image capture from GIGE camera

%x=videoinput('winvideo', 1);
%x=videoinput('winvideo', 2, 'RGB32_2448x2048');
g = gigecam;
preview(g);

img = snapshot(g);
%img=getsnapshot(x);
img=img(:,:,1);
subplot(2,2,1);
imshow(img);
title('Live Image');


%imwrite(img, ['camera.png']);
img = imlocalbrighten(img);
img = imsharpen(img);
%img = imadjust(img);
%img = imbinarize(img);

%I = imread('c.bmp');
%I=rgb2gray(img);

imwrite(img(700:1400,100:800),('testSet\\first\\live_image.png'));
I_cut = imread('testSet\\first\\live_image.png');
subplot(2,2,2);
imshow(I_cut);
title('Live cropped Image');

subplot(2,2,3);
I_cut = imbinarize(I_cut);
imshow(I_cut);
title('Binary');

[centers,radii]=imfindcircles(I_cut,[10 20],'ObjectPolarity','dark','Sensitivity',0.85)

% Crop to 41x41 px (approximately double the outlet radius)
startpoint=round(centers);
west=startpoint(1)-25;
east=startpoint(1)+25;
north=startpoint(2)-25;
south=startpoint(2)+25;
i=1;

while j<=2

    I = snapshot(g);
    I=I(700:1400,100:800);
    imwrite(I(north:south,west:east),['testSet\\first\\Image.png']);

IZ=I(north:south,west:east);
subplot(2,2,4);
imshow(IZ);

    imwrite(IZ,['Images\\correct_cut\\round_',num2str(i),'.png'])
    %fname=['Images\\image_', num2str(i)];
    %imwrite(IZ,fname,'jpg');
title('binary croped circle');

imageFolder2 = fullfile('testSet\\first');
imds2 = imageDatastore(imageFolder2, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

testSet = imds2;

% Apply trained Classifier on one Test Image
testImage = readimage(testSet,1);
% replace testSet image with direct img

testLabel = testSet.Labels(1);
% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

% make a prediction using the classifier
predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')

switch predictedLabel
        case 'full'
        %j=j+1
        case 'empty'
        j=0
end

if(j==3)
    beep
    f=msgbox('STOP Gluing','Error','error');
    
    break
end
i=i+1;
  
end
%% Load images from subfolders

% This code cuts all the images based on the first image data

path=sprintf('testSet\\first\\');
picpath=[path '*PNG'];
z= dir (picpath);

%% Find outlet and crop image
% Danger! The naming array is not sorted by size, but by the individual characters.
% This means that 1000 is less than 900

file=z(1).name;             % Call filename
filepath=[path file];       % Merge for file retrieval
X{1}=imread(filepath);      % Read image
%X{1}=X{1}(:,:,:);       % Rough clipping helps to find an outlet
%X{1}=X{1}(60:500,150:700,:); 

[centers,radii]=imfindcircles(X{1},[10 25],'ObjectPolarity','dark','Sensitivity',0.85);

%% Beschneiden auf 41x41 px (ungefaehr doppelter Auslassradius)
startpoint=round(centers);
west=startpoint(1)-20;
east=startpoint(1)+20;
north=startpoint(2)-20;
south=startpoint(2)+20;

path=sprintf('testSet\\raw\\');
picpath=[path '*PNG'];
z= dir (picpath); 

while j<=1
    %img=getsnapshot(x);
    %img = snapshot(g);
    %fname=('testSet\\raw\\image.png');
    %imwrite(img,fname);
    
% Cropping all images in the relevant folder        
for k=1:numel(z)
    file = z(k).name;
    filepath = [path file];
    X{k} = imread(filepath);   
    X{k} = rgb2gray(X{k});
    %X{k} = imbinarize(X{k});

    imwrite(X{k}(north:south,west:east,:),['testSet\\round','.png'])
   
    subplot(2,4,k+2);
    imshow('testSet\\round.png');
%% Pass CNN image features to trained classifier

imageFolder2 = fullfile('testSet');
imds2 = imageDatastore(imageFolder2, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

testSet = imds2;

% Apply trained Classifier on one Test Image
testImage = readimage(testSet,1);
% replace testSet image with direct img

testLabel = testSet.Labels(1);
% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

% make a prediction using the classifier
predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')

switch predictedLabel
        case 'full'
        j=j+1;
        case 'empty'
        j=0;
end

if(j==2)
    break
end

end
    %delete('testSet\\raw\\image.png');
    %delete('testSet\\first\\image.png');
    %delete('testSet\\round.png');
    pause(0.2);

end






