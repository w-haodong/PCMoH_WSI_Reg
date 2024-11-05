function [cleanedPoints1, cleanedPoints2, finalH] = RIFT(im1, im2)

% RIFT feature detection and description
[des_m1,des_m2] = RIFT_no_rotation_invariance(im1,im2,4,6,96);

% nearest matching
[indexPairs,~] = matchFeatures(des_m1.des,des_m2.des,'MaxRatio', 1, 'MatchThreshold', 100);
matchedPoints1 = des_m1.kps(indexPairs(:, 1), :);
matchedPoints2 = des_m2.kps(indexPairs(:, 2), :);
[matchedPoints2,IA]=unique(matchedPoints2,'rows');
matchedPoints1=matchedPoints1(IA,:);

%outlier removal
H=FSC(matchedPoints1,matchedPoints2,'affine',2);
Y_=H*[matchedPoints1';ones(1,size(matchedPoints1,1))];
Y_(1,:)=Y_(1,:)./Y_(3,:);
Y_(2,:)=Y_(2,:)./Y_(3,:);
E=sqrt(sum((Y_(1:2,:)-matchedPoints2').^2));
inliersIndex=E<3;
cleanedPoints1 = matchedPoints1(inliersIndex, :);
cleanedPoints2 = matchedPoints2(inliersIndex, :);
finalH = H;