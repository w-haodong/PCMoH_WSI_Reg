function [cleanedPoints1, cleanedPoints2, finalH] = RIFT2(im1, im2)

[key1,m1,eo1] = FeatureDetection(im1,4,6,5000);
[key2,m2,eo2] = FeatureDetection(im2,4,6,5000);


kpts1 = kptsOrientation(key1,m1,1,96);
kpts2 = kptsOrientation(key2,m2,1,96);


des1 = FeatureDescribe(im1,eo1,kpts1,96,6,6);
des2 = FeatureDescribe(im2,eo2,kpts2,96,6,6);


[indexPairs,matchmetric] = matchFeatures(des1',des2','MaxRatio',1,'MatchThreshold', 100);
kpts1 = kpts1'; kpts2 = kpts2';
matchedPoints1 = kpts1(indexPairs(:, 1), 1:2);
matchedPoints2 = kpts2(indexPairs(:, 2), 1:2);
[matchedPoints2,IA]=unique(matchedPoints2,'rows');
matchedPoints1=matchedPoints1(IA,:);

H=FSC(matchedPoints1,matchedPoints2,'similarity',3);
Y_=H*[matchedPoints1';ones(1,size(matchedPoints1,1))];
Y_(1,:)=Y_(1,:)./Y_(3,:);
Y_(2,:)=Y_(2,:)./Y_(3,:);
E=sqrt(sum((Y_(1:2,:)-matchedPoints2').^2));
inliersIndex=E<3;
cleanedPoints1 = matchedPoints1(inliersIndex, :);
cleanedPoints2 = matchedPoints2(inliersIndex, :);
finalH = H;


