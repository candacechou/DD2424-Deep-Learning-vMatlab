
% function [X, Y, y] = LoadBatch(filename)
% . input :
% .         filename
%   output:
%           X   = dxN
%           X contains the image pixel data, has entries between 0 and 1. N is the number of images (10000) 
%           and d the dimensionality of each image (3072=32×32×3)
%           Y   = K×N 
% .         (K  = # of labels = 10) 
%           y   = vector of length N containing the label for each image. 
function [X, Y, y] = LoadBatch(filename)


    d       = 32 * 32 * 3;
    N       = 10000;
    K       = 10;
    X       = zeros(d,N);
    Y       = zeros(K,N);
    y       = zeros(1,N);
 
    A       = load(filename);
    I       = reshape(A.data',32,32,3,10000);
    X       = reshape(I,32*32*3,10000);
    X       = double(X)./255;
    labels  = A.labels;
    %%% one-hot representation for Y
    for i = 1:10 
        row         = find(labels == i-1);
        Y(i,row)    = 1;
    end
    y       = labels + 1;
    
   
end
