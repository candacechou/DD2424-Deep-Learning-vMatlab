% function [Wstar, bstar,l,t,eta] = MiniBatchGD(X, Y, n_batch,eta,eta_min,eta_max,ns,n_epochs, W, b, lambda)
%      inputs:
%         X = dxn
%         Y = Kxn
%         n_batch = 1x1
%         eta = 1x1
%         eta_min = 1x1
%         eta_max = 1x1
%         ns = 1x1
%         n_epochs = 1x1
%         W = cell(1,2)
%         b = cell(1,2)
%         l = 1x1
%         t = 1x1
%       outputs:
%         Wstar: cell(1,2)
%         bstar: cell(1,2)
%         l = 1x1
%         t = 1x1
%        eta = 1x1
function [Wstar, bstar,l,t,etas,t_cost,t_loss,t_acc,v_cost,v_loss,v_acc] = MiniBatchGD(X, Y, n_batch,eta,eta_min,eta_max,ns, W, b, lambda,l,t,t_cost,t_loss,t_acc,v_cost,v_loss,v_acc)
etas = eta;
[K,n] = size(Y);
[d,n] = size(X);
Wstar = W;
bstar = b;
% s = rng(400);
%rng(s);
rand_i = randperm(n);
%rand_i = 1:n;

shuffle_x = X(:,rand_i);
shuffle_y = Y(:,rand_i);
for j=1:n/n_batch
    j_start = (j-1) * n_batch + 1;
    j_end = j * n_batch;
    inds = j_start:j_end;
    Xbatch = shuffle_x(:, j_start:j_end);
    Ybatch = shuffle_y(:, j_start:j_end);
    [h,P] = EvaluateClassifier(Xbatch, Wstar, bstar);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch,P,h,Wstar,bstar,lambda);
    if (t >= 2 * l * ns) && (t < (2*l + 1) * ns)
        etas_unit = eta_min + ((t - 2 * l * ns) / ns) * (eta_max - eta_min);
    elseif (t>=(2*l + 1) * ns) && t <(2*(l+1)*ns)
        etas_unit = eta_max - ((t - (2 * l+1) * ns) / ns) * (eta_max - eta_min);
    end
    
    etas(t,1) = etas_unit;
    Wstar{1} = Wstar{1} - etas_unit * grad_W{1};
    bstar{1} = bstar{1} - etas_unit * grad_b{1};
    Wstar{2} = Wstar{2} - etas_unit * grad_W{2};
    bstar{2} = bstar{2} - etas_unit * grad_b{2};
    t = t + 1;
    
    if rem(t,2*ns) == 0
        l = l+1;
    end
    if rem(t,10) == 0
        t_cost(i,1) = ComputeCost(trainX,trainY,W,b,lambda);
        t_loss(i,1) = ComputeLoss(trainX,trainY,trainy,W,b);
        t_acc(i,1) = ComputeAccuracy(trainX,trainy,W,b);
        v_cost(i,1) = ComputeCost(ValX,ValY,W,b,lambda);
        v_loss(i,1) = ComputeLoss(ValX,ValY,Valy,W,b);
        v_acc(i,1) = ComputeAccuracy(ValX,Valy,W,b);
        yline(i,1) = t;
        i = i+1;
    end
   
end
end
