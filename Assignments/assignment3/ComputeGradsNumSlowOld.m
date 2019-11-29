function [grad_b, grad_W] = ComputeGradsNumSlowOld(X, Y, net, lambda, h,k_layers)
W      = net.W;
b      = net.b;
grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        net1   = net;
        net1.b = b_try;
        c1 = ComputeCost(X, Y, net1,lambda,k_layers);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        net1.b = b_try;
        c2 = ComputeCost(X, Y, net1, lambda,k_layers);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        net2  = net;
      
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        net2.W = W_try;
        c1 = ComputeCost(X, Y, net2, lambda,k_layers);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        net2.W = W_try;
        c2 = ComputeCost(X, Y, net2, lambda,k_layers);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end