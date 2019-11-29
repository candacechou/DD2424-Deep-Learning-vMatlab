%% function P = EvaluateClassifier(X, W, b)
%     evaluates the network function,
%     input :
%           X = dxn   each column of X corresponds to an image and it has size d×n.
%           W = Kxd 
%           b = Kx1 
%     output :
%           P = Kxn  each column of P contains the probability for each label for the image in the corresponding column of X. P has size K×n.
% 

function P = EvaluateClassifier(X,W,b)
        [d,n]   = size(X);
        [k,d]   = size(W);
        P       = zeros(k,n);
        bm      = repmat(b,1,n);
        P       = W * double(X) + bm;
        P       = exp(P);
        Psum    = sum(P,1);
        Psum    = repmat(Psum,k,1);
        P       = P ./Psum;       
end