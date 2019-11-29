% function J = ComputeCostSVM(X, Y, W, b, lambda)      
%  computes the cost function given by equation (5) for a set of images.
%  
%  input:
%         X         = dxn
%         Y         = Kxn
%         W         = Kxd
%         b         = Kx1
%         lambda    = 1x1
%         C         = 1x1
%  output:
%         J         = 1x1 scalar corresponding to the sum of the loss of the network’s predictions for the images in X relative to the 
%         ground truth labels and the regularization term on W.
%     

function J = ComputeCostSVM(X, Y, W, b, lambda,C)  
        J       = 0;
        
        P       = EvaluateClassifierSVM(X,W,b,Y);
        [d,D]   = size(X);
        %J = C * sum(sum(P)) + lambda * sum(W.^2,'all');
        J       = 1/D * (sum(sum(P)-1)) + lambda * sum(W.^2,'all');
        
end