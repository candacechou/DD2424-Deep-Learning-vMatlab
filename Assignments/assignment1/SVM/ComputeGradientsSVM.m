
% function [grad_W, grad_b] = ComputeGradientsSVM(X, y, b, W, lambda)
% the function that evaluates, for a mini-batch, the gradients of the cost function w.r.t. W and b
%     input :
%         X         = dxn
%         y         = 1x1
%         b         = Kx1
%         W         = Kxd
%         lambda    =  1x1
%     output : 
%         grad_w    = Kxd   the gradient matrix of the cost J relative to W 
%         grad_b    = Kx1   the gradient vector of the cost J relative to b
function [grad_w, grad_b] = ComputeGradientsSVM(X,y,Y,b,W,lambda)
    [K,n]       = size(X);
    [K,d]       = size(W);
    grad_w      = zeros(K,d);
    grad_b      = zeros(K,1);

    P           = EvaluateClassifierSVM(X,W,b,Y);
    for i=1:n
        xi      = X(:,i);
        Pi      = P(:,i);
        gi      = repmat(xi',K,1);
        gb      = zeros(K,1);
        a       = find(Pi > 0 & Pi~=1);
        gb(a)   = 1;
        a       = find(Pi == 1);
        gi(a,:) = -gi(a,:)*(length(find(Pi > 0&Pi~=1)));
        gb(a)   = -length(find(Pi>0&Pi~=1));
        a       = find(Pi == 0);
        gi(a,:) = 0;
        gb(a)   = 0;
       
        grad_w  = grad_w + gi;
        grad_b  = grad_b + gb;
    end
 grad_w     = grad_w./n + 2*lambda*W;
 grad_b     = grad_b./n;
    end
        
%     temp      = W * double(X) + bm; 
%     tempy     = repmat(y',K,1);
%     tempy     = double(tempy);
%     check     = temp .* tempy;
%     dx        = find(check< 1)
    

