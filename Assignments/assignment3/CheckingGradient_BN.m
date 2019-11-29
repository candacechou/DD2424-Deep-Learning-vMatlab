function [] = CheckingGradient_BN(net,grad_BN,k_layers,h)
    for i = 1:k_layers
        fprintf('the %d th layers\n',i);
        if i <= k_layers-1
            
            Wa = abs(net.Grad_W{i,1} - grad_BN.W{i,1});
            fprintf('W');
            find(Wa > h)
            Wb = abs(net.Grad_b{i,1} - grad_BN.b{i,1});
            fprintf('b');
            find(Wb > h)
            Wg = abs(net.Grad_gm{i,1} - grad_BN.gammas{i,1});
            fprintf('gamma');
            find(Wg > h)
            Wbeta = abs(net.Grad_bt{i,1} - grad_BN.betas{i,1});
            fprintf('betas');
            find(Wbeta > h)
        else
            fprintf('the %d th layers\n',i);
            Wa = abs(net.Grad_W{i,1} - grad_BN.W{i,1});
            fprintf('W');
            find(Wa > h)
            Wb = abs(net.Grad_b{i,1} - grad_BN.b{i,1});
            fprintf('b');
            find(Wb > h)
        end
    end
end
