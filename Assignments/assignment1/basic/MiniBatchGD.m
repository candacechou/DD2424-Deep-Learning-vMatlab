% function [Wstar, bstar] = MiniBatchGD(X, Y, n_batch,eta,n_epochs, W, b, lambda)
%      inputs:
%         X         = dxn
%         Y         = Kxn
%         n_batch   = 1x1
%         eta       = 1x1
%         n_epochs  = 1x1
%         W         = Kxd
%         b         = Kx1
%       outputs:
%         Wstar     = Kxd
%         bstar     = Kx1
function [Wstar, bstar] = MiniBatchGD(X, Y, n_batch,eta,n_epochs, W, b, lambda)
[K,n]       = size(Y);
[d,n]       = size(X);
Wstar       = W;
bstar       = b;
% s         = rng(400);
%rng(s);
rand_i      = randperm(n);
%rand_i     = 1:n;

shuffle_x   = X(:,rand_i);
shuffle_y   = Y(:,rand_i);
for j=1:n/n_batch
    j_start             = (j-1) * n_batch + 1;
    j_end               = j * n_batch;
    inds                = j_start:j_end;
    Xbatch              = shuffle_x(:, j_start:j_end);
    Ybatch              = shuffle_y(:, j_start:j_end);
    P                   = EvaluateClassifier(Xbatch, Wstar, bstar);
    [grad_W, grad_b]    = ComputeGradients(Xbatch, Ybatch,P,Wstar,lambda);
    %[grad_b, grad_W]   = ComputeGradsNum(Xbatch, Ybatch, Wstar, b, lambda, 1e-6);
    Wstar               = Wstar - eta * grad_W;
    bstar               = bstar - eta * grad_b;
    % for stochastic
end
end
