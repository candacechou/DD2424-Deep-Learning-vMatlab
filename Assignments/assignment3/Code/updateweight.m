% function net = updateweight(net,eta,k_layers,BN)
% update the weight
function net = updateweight(net,eta,k_layers,BN)
    for i=1:k_layers
        net.W{i}      = net.W{i} - eta * net.Grad_W{i};
        net.b{i}      = net.b{i} - eta * net.Grad_b{i};
        if BN == 1 && i<k_layers
            net.gammas{i}        = net.gammas{i} - eta * net.Grad_gm{i};
            net.betas{i}         = net.betas{i} - eta * net.Grad_bt{i};         
    end
end