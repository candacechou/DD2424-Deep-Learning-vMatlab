%   function [net] = InitializeParameter(X,k_layers,hidden_layers)
%     
%    To initialize the parameters for 2 layer neural network
%     Input:
%         X = dxn
%         k_layers = 1x1
%         hidden_layers = 1x(k_layers-1)
%       
%     output:
%      net : a defined class that has been written in the other file called
%      net_parms

function [net] = InitializeParameter(X,k_layers,hidden_layers)
    [d,n]           = size(X);
    net             = net_params;
    net.use_bn      = 0;
    net.W           = cell(k_layers,1);
    net.Grad_W      = cell(k_layers,1);
    net.Grad_b      = cell(k_layers,1);
    net.b           = cell(k_layers,1);
    Xinit           = sqrt(2)/sqrt(d);
    net.W{1,1}      = Xinit * randn(hidden_layers(1),d);
    net.Grad_W{1,1} = zeros(hidden_layers(1),d);
    net.b{1,1}      = zeros(hidden_layers(1),1);
    net.Grad_b{1,1} = zeros(hidden_layers(1),1);
    for i=1:k_layers-1
        if i ~= k_layers-1
            Xinit               = sqrt(2)/sqrt(hidden_layers(i));
            net.W{i+1,1}        = Xinit * randn(hidden_layers(i+1),hidden_layers(i));
            net.b{i+1,1}        = zeros(hidden_layers(i+1),1);
            net.Grad_W{i+1,1}   = zeros(hidden_layers(i+1),hidden_layers(i));
            net.Grad_b{i+1,1}   = zeros(hidden_layers(i+1),1);
        else
            Xinit               = sqrt(2)/sqrt(hidden_layers(i));
            net.W{i+1,1}        = Xinit * randn(10,hidden_layers(i));
            net.b{i+1,1}        = zeros(10,1);
            net.Grad_W{i+1,1}   = zeros(10,hidden_layers(i));
            net.Grad_b{i+1,1}   = zeros(10,1);
        end
    end
end