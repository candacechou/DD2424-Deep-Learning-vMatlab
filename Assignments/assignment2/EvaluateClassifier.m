%% function [P,h] = EvaluateClassifier(X, W,b)
%     evaluates the network function,
%     input :
%           X  =  dxn   each column of X corresponds to an image and it has size d×n.
%           W = cell(1,2)
%             W{1} = W1
%             W{2} = W2
%             b = cell(1,2)
%             b{1} = b1
%             b{2} = b2
%             ((W1 = Kxd 
%               b1 = Kx1 
%               W2 = kXk
%               b2 = Kx1))
%     output :
%           P = Kxn  each column of P contains the probability for each label for the image in the corresponding column of X. P has size K×n.
%           h = kxn  for calculating the gradient value
function [h,P] = EvaluateClassifier(X, W,b)
        [d,n] = size(X);
        [m,d] = size(W{1});
        k=10;
        P = zeros(k,n);
        b1_m  = repmat(b{1},1,n);
        %%%% The first layer
        s1 = W{1} * X + b1_m;
        h = max(0,s1);  %%%% kxn
        b2_m = repmat(b{2},1,n);
        %%%% The second layer
        P = W{2} * h + b2_m;
        %%%% softmax
        P = exp(P);
        Psum = sum(P,1);
        Psum = repmat(Psum,k,1);
        P = P ./Psum; 
        clear Psum;clear s1;clear b2_m; clear b1_m;
       
        
end