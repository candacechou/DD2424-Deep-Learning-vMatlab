% function [net] = ComputeGradients(X, Y, net , lambda,k_layers,P,h)
% the function that evaluates, for a mini-batch, the gradients of the cost function w.r.t. W and b
%     input :
%         X         = dxn
%         Y         = Kxn
%         net       = net
%         lambda    = 1x1
%         k_layers  = 1x1
%         P         = nx10
%         h         = cell{k_layers,1}
%     output : 
%          net      = net -> grad_W and grad_b only change in the net,
%          there are parameters in the class net_params

function [net] = ComputeGradients(X, Y, net , lambda,k_layers,P,h)
    [K,n]       = size(Y);
    [d,n]       = size(X);
    %[P,h,s]     = EvaluateClassifier(X,net,k_layers);
    g           = -(Y-P);
    for i = k_layers:-1:2
        net.Grad_b{i,1}   = sum(g,2)./n;
        net.Grad_W{i,1}   = g * h{i-1,1}'./n + 2 * lambda * net.W{i,1};
        g                 = net.W{i,1}' * g;
        indX              = h{i-1,1};
        indX(indX < 0)    = 0;
        indX(indX > 0)    = 1;
        g                 = g .* indX;
    end
    %%% finally :
    
    net.Grad_W{1,1}       = g * X'./n;
    net.Grad_b{1,1}       = sum(g,2)./n;
        
        
        
end