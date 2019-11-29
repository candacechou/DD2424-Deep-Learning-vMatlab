% function J = ComputeCost(X, Y, W, b, lambda)      
%  computes the cost function given by equation (5) for a set of images.
%  
%  input:
%         X         = dxn
%         Y         = Kxn
%         W         = Kxd
%         b         = Kx1
%         lambda    = 1x1
%  output:
%         J         = 1x1 scalar corresponding to the sum of the loss of the network’s predictions for the images in X relative to the 
%         ground truth labels and the regularization term on W.
%     

function J = ComputeCost(X, Y, W, b, lambda)  
        J       = 0;
        P       = EvaluateClassifier(X,W,b);
        l       = -log(Y'* P); %% nxn
        [d,D]   = size(X);
        J       = (1/D) * sum(diag(l),'all') + lambda * sum(W.^2,'all');
        
end