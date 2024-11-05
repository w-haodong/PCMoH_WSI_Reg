function [cleanedPoints1, cleanedPoints2, finalH] = sRIFD(im1, im2)
% sRIFD feature detection and description
[des_m1,des_m2,or1,or2] =  sRIFD_rotation_invariance(im1,im2,6,6);
temp=0;
for ii=1:24
    % nearest matching
    des1=des_m1.des(1,:,:);
    des1=squeeze(des1)';
    des2=des_m2.des(ii,:,:);
    des2=squeeze(des2)';

    [indexPairs,matchmetric] = matchFeatures(des1,des2,'MaxRatio',1,'MatchThreshold', 100);
    matchedPoints1 = des_m1.kps(indexPairs(:, 1), :);
    matchedPoints2 = des_m2.kps(indexPairs(:, 2), :);
    [matchedPoints2,IA]=unique(matchedPoints2,'rows');
    matchedPoints1=matchedPoints1(IA,:);


    %outlier removal
    H=FSC(matchedPoints1, matchedPoints2,'affine',2);
    Y_=H*[matchedPoints1';ones(1,size(matchedPoints1,1))];
    Y_(1,:)=Y_(1,:)./Y_(3,:);
    Y_(2,:)=Y_(2,:)./Y_(3,:);
    E=sqrt(sum((Y_(1:2,:)-matchedPoints2').^2));
    inliersIndex=E<3;

    if sum(inliersIndex)>temp
        temp = sum(inliersIndex);
        finInliners=inliersIndex;
        finalMatchedPoints1=matchedPoints1;
        finalMatchedPoints2=matchedPoints2;
        finalH = H;
    end
end

cleanedPoints1 = finalMatchedPoints1(finInliners, :);
cleanedPoints2 = finalMatchedPoints2(finInliners, :);


