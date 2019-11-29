
% function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
% the function that evaluates, for a mini-batch, the gradients of the cost function w.r.t. W and b
%     input :
%         X         = dxn
%         Y         = Kxn
%         P         = Kxn
%         W         = Kxd
%         lambda    = 1x1
%     output : 
%         grad_w    = Kxd   the gradient matrix of the cost J relative to W 
%         grad_b    = Kx1   the gradient vector of the cost J relative to b
function [grad_w, grad_b] = ComputeGradients(X,Y,P,W,lambda)
    [K,n]       = size(Y);
    [K,d]       = size(W);
    grad_w      = zeros(K,d);
    grad_b      = zeros(K,1);
    g           = -(Y-P)';
    X           = double(X);
    grad_b      = sum(g,1)'./n;  
    grad_w      = g' * X'./n + 2 * lambda * W;   
end