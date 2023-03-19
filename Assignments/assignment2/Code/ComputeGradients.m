
% function [grad_W, grad_b] = ComputeGradients(X, Y, P,h, W1,b1,W2,b2, lambda)
% the function that evaluates, for a mini-batch, the gradients of the cost function w.r.t. W and b
%     input :
%         X = dxn
%         Y = Kxn
%         P = Kxn
%         h = kxn
%         W = cell(1,2)
%         b = cell(1,2)
%         ((W1= Kxd
%         b1 = Kx1
%         W2 = kxk
%         b2 = kx1))
%         lambda = 1x1
%     output : 
%          grad_W = cell(1,2)
%          grad_b = cell(1,2)
%         ((grad_W1 = Kxd    
%         grad_b1 = Kx1   
%         grad_W2 = KxK
%         grad_b1 = Kx1))

function [grad_W, grad_b] = ComputeGradients(X, Y, P,h, W,b, lambda)
    [K,n] = size(Y);
    [d,n] = size(X);
    grad_W = cell(1,2);
    grad_b = cell(1,2);

    %%%% gradient of W2 and b2
    g = - (Y - P);
    grad_W{2} = g * h'./n + 2 * lambda * W{2} ;
    grad_b{2} = sum(g,2)./n;
    %%%% gradient of W1 and b1
    g_batch = W{2}'* g; %%% kxn
    ind_h = h > 0; %%%% kxn
    g_batch = g_batch .* ind_h; %%%% kxn
    grad_W{1} = g_batch * X'./n + 2 * lambda * W{1};
    grad_b{1} = sum(g_batch,2)./n;
    clear g;
    clear g_batch;
end