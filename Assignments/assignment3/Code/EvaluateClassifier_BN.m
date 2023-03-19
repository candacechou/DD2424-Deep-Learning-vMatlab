%% function [P,Xhat_batch,s,s_hat,net] = EvaluateClassifier_BN(X,net,k_layers,t,cost)
%     evaluates the network function,
%     input :
%           X                 = dxn                each column of X corresponds to an image and it has size d×n.
%           net               = net
%           k_layers          = 1x1
%           t                 = 1x1 to check if we need to initialize ave_u
%           cost              = 1x1 to check if this is from cost function
%           or not
%     output :
%           P                 = Kxn                each column of P contains the probability for each label for the image in the corresponding column of X. P has size K×n.
%           Xhat_batch        = cell(k_layes-1,1)  for calculating the gradient value
%           s                 = cell(k_layers-1,1)
%           s_hat             = cell(k_layers-1,1)
%           net               = net
function [P,Xhat_batch,s,s_hat,net] = EvaluateClassifier_BN(X,net,k_layers,t,cost)
        [d,n]            = size(X);
        k                = 10;
        x_temp           = X;
        j                = k_layers - 1;
        Xhat_batch       = cell(k_layers,1);
        Xhat_batch{1,1}  = x_temp;
        s                = cell(j,1);
        s_hat            = cell(j,1);
        for i=1:k_layers-1
            %b_m             = repmat(net.b{i,1},1,n);
            s{i,1}              = net.W{i,1} * x_temp + net.b{i,1};
            %%%% not sure assuming make average on each dimension
            net.un_mu{i,1}      = sum(s{i,1},2)/size(s{i,1},2);
            %net.un_mu{i,1}  = mean(s{i,1},2);
            net.un_v{i,1}       = sum((s{i,1} -net.un_mu{i,1}).^2,2)./size(s{i,1},2);
%             net.ave_mu{i,1}     = net.un_mu{i,1};
%             net.ave_v{i,1}      = net.un_v{i,1};
            
            %%%% Batch Normalization
            if cost == 0
                s_hat{i,1}          = Batch_Normalization(s{i,1},net.un_mu{i,1},net.un_v{i,1});
                if t == 1
                    net.ave_mu{i,1} = net.un_mu{i,1};
                    net.ave_v{i,1}  = net.un_v{i,1};
                else
            %%%% do the moving average
                    net.ave_mu{i,1}     = net.alpha * net.ave_mu{i,1} + (1 - net.alpha) * net.un_mu{i,1};
                    net.ave_v{i,1}      = net.alpha * net.ave_v{i,1} + (1 - net.alpha) * net.un_v{i,1};
                end
            else
                s_hat{i,1}          = Batch_Normalization(s{i,1},net.ave_mu{i,1},net.ave_v{i,1});
            end
            s_shave             = repmat(net.gammas{i},1,n) .*  s_hat{i,1} + net.betas{i};
            x_temp              = max(0,s_shave);
            Xhat_batch{i+1,1}   = x_temp;
        end
        s_temp            = net.W{k_layers,1} * x_temp + net.b{k_layers,1};
        P                 = exp(s_temp);
        Psum              = sum(P,1);
        Psum              = repmat(Psum,k,1);
        P                 = P ./Psum; 
       
        
        
        
end