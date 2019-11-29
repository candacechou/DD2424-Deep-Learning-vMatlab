% function [net] = ComputeGradients_BN(X, Y, net , lambda,k_layers,P,Xhat_batch,s,s_hat)
% the function that evaluates, for a mini-batch, the gradients of the cost function w.r.t. W and b
%     input :
%         X                  = dxn
%         Y                  = Kxn
%         net                = net
%         lambda             = 1x1
%         k_layers           = 1x1
%         P                  = nx10
%         Xhat_batch         = cell{k_layers-1,1}
%         s                  = cell{k_layers-1,1}
%         s_hat              = cell{k_layers-1,1}
%     output : 
%          net              = net -> grad_W and grad_b only change in the net,
%          there are parameters in the class net_params

function [net] = ComputeGradients_BN(X, Y, net , lambda,k_layers,P,Xhat_batch,s,s_hat)
    [K,n]                   = size(Y);
    [d,n]                   = size(X);
    g                       = -(Y-P);
    net.Grad_b{k_layers}    = sum(g,2)./n;
    net.Grad_W{k_layers}    = g * Xhat_batch{k_layers}'./n + 2 * lambda * net.W{k_layers,1};
    g                       = net.W{k_layers}' * g;
    indX                    = Xhat_batch{k_layers};
    indX(indX < 0)          = 0;
    indX(indX > 0)          = 1;
    g                       = g .* indX;
    for i = k_layers-1:-1:1
        
            [col,n]                 = size(g);
%             I_n                     = ones(n);
            net.Grad_gm{i}          = sum(g .* s_hat{i},2)./n;
            net.Grad_bt{i}          = sum(g,2)./n;
            g                       = g.*repmat(net.gammas{i},1,n);
            %%%% batchNormBackPass
            g                       = BatchNormBackPass(g,s{i},net.un_mu{i},net.un_v{i});
            %%%% calculate the gradient of bias and weights
            
            net.Grad_b{i}           = sum(g,2)./n;
            net.Grad_W{i}           = g * Xhat_batch{i}'./n + 2 * lambda * net.W{i};
            %%%% calculate new G_batch
            if i > 1
                g                       = net.W{i}' * g;
                indX                    = Xhat_batch{i};
                indX(indX < 0)          = 0;
                indX(indX > 0)          = 1;
                g                       = g .* indX;
                
            end
        end
    %%% finally the first layer of gradient W and b :
    
%     net.Grad_W{1}           = g * X'./n;
%     net.Grad_b{1}           = sum(g,2)./n;
%         
        
        
end