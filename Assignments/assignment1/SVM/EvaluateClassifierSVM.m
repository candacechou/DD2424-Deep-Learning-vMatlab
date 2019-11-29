%% function P = EvaluateClassifierSVM(X, W, b,Y)
%     evaluates the network function,
%     input :
%           X = dxn   each column of X corresponds to an image and it has size d×n.
%           W = Kxd 
%           b = Kx1
%           Y = kxn
%           
%     output :
%           P = Kxn  each column of P contains the probability for each label for the image in the corresponding column of X. P has size K×n.
% 

function P = EvaluateClassifierSVM(X,W,b,Y)
        [d,n]       = size(X);
        [k,d]       = size(W);
        P           = zeros(k,n);
        bm          = repmat(b,1,n);
        temp        = W * double(X) + bm;
        temp_w      = Y .* temp;
        c           = find(temp_w ~=0);
        xd          = temp(c);
        temp_x      = repmat(xd',k,1);
        ss          = temp - temp_x + 1;
        
        P           = max(0,ss);
end