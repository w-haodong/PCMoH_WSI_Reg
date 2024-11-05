clear
clc
warning('off')

% choose algorithm (RIFT RIFT2 sRIFD RIFT_OUR)
affine_fun = "RIFT_OUR";
% image path
img_path = "./test_images/img1/";
% choose Level (1-6)
level = 5;


% add code path
addpath ./RIFT/ ./RIFT2/ ./RIFT_OUR/ ./sRIFD/

str1 = img_path + "source_level_" + num2str(level) + ".jpg";
str2 = img_path + "target_level_" + num2str(level) + ".jpg";
im1 = im2uint8(imread(str1));
im2 = im2uint8(imread(str2));

if affine_fun == "RIFT_OUR"
    system('.\RIFT_OUR\RIFT_OUR.exe '+str1+" "+str2);
    load("res.mat");
else
    [cleanedPoints1, cleanedPoints2, finalH] = eval([affine_fun+"(im1, im2);"]);
end
disp('Show matches')
% Show results
figure;
showMatchedFeatures(im1, im2, cleanedPoints1, cleanedPoints2, 'montage');

disp('registration result')
% registration
image_fusion(im2,im1,double(finalH));