% function [W1,W2,b1,b2, lambda] = InitializeParameter(X,k_num,std_1,mean_1,std_2,mean_2,b_1,b_2,node_num,lbda)
%     
%    To initialize the parameters for 2 layer neural network
%     Input:
%         X = dxn
%         k_num = 1x1
%         std_1 = 1x1
%         mean_1 = 1x1
%         std_2 = 1x1
%         mean_2 = 1x1
%         b_1 = 1x1
%         b_2 =1x1   
%         node_num = 1x1
%         l_min = 1x1
%         l_max = 1x1
%     output:
%           W = cell(1,2)
%             W(1) = W1
%             W(2) = W2
%         ((W1 = k_num x d
%           b1 = k_num x 1
%           W2 = K_num x d
%           b2 = k_mun x 1))
%         lambda = 1x1
function [W,b, lambda] = InitializeParameter(X,k_num,std_1,mean_1,std_2,mean_2,l_min,l_max,m)
[d,n] = size(X);
%m = 1/std_2^2;
l = l_min + (l_max - l_min)*rand(1, 1); 
lambda = 10^l;
W1 = zeros(m,d);
W1 = std_1 * randn(m,d) + mean_1;
b1 = zeros(m,1);
W2 = zeros(k_num,m); %%%% not sure
W2 = std_2 * randn(k_num,m') + mean_2;
b2 = zeros(k_num,1);
W = cell(1,2);
W = {W1,W2};
b = {b1,b2};
end

