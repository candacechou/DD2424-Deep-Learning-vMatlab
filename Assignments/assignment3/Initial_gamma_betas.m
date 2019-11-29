%   function [net] = Initial_gamma_betas(net,k_layers,hidden_layers,batch_size,alpha)
%     
%    To initialize the parameters for 2 layer neural network
%     Input:
%         net           = net
%         k_layers      = 1x1
%         hidden_layers = 1x(k_layers-1)
%         batch_size    = 1x1
%         
%     output:
%      net : a defined class that has been written in the other file called
%      net_parms

function [net] = Initial_gamma_betas(net,k_layers,hidden_layers,batch_size,alpha)
    net.alpha       = alpha;
    net.betas       = cell(k_layers-1,1);
    net.gammas      = cell(k_layers-1,1);
    for i=1:k_layers-1
        net.betas{i}    = zeros(hidden_layers(i),1);
        net.gammas{i}   = ones(hidden_layers(i),1);
    end
end