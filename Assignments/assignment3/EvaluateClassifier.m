%% function [P,h] = EvaluateClassifier(X,net,k_layers)
%     evaluates the network function,
%     input :
%           X        =  dxn   each column of X corresponds to an image and it has size d×n.
%           net      = net
%           k_layers = 1x1
%     output :
%           P        = Kxn             each column of P contains the probability for each label for the image in the corresponding column of X. P has size K×n.
%           h        = cell(k_layes-1,1)  for calculating the gradient value
%           s_sum    = cell(k_layers,1)


function [P,h,s_sum] = EvaluateClassifier(X,net,k_layers)
        [d,n]   = size(X);
        k       = 10;
        x_temp  = X;
        j       = k_layers - 1;
        h       = cell(j,1);
        s_sum   = cell(j,1);
        for i=1:k_layers-1
            b_m        = repmat(net.b{i,1},1,n);
            s          = net.W{i,1} * x_temp + b_m;
            x_temp     = max(0,s);
            h{i,1}     = x_temp;
            s_sum{i,1} = s;
        end
        b_m               = repmat(net.b{k_layers,1},1,n);
        s                 = net.W{k_layers,1} * x_temp + b_m;
        s_sum{k_layers,1} = s;
        P                 = exp(s);
        Psum              = sum(P,1);
        Psum              = repmat(Psum,k,1);
        P                 = P ./Psum; 
            
        
        
end