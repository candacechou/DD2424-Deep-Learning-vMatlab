% function J = ComputeCost(X, Y, W, b, lambda)      
%  computes the cost function given by equation  for a set of images.
%  
%  input:
%         X = dxn
%         Y = Kxn
%         W = cell{1,2}
%         b = cell{1,2}
%         lambda = 1x1
%  output:
%         J = 1x1 scalar corresponding to the sum of the loss of the network’s predictions for the images in X relative to the 
%         ground truth labels and the regularization term on W.

function J = ComputeCost(X, Y, W, b, lambda) 
        J = 0;
        [h,P] = EvaluateClassifier(X,W,b);
        l = -log(Y'* P); %% nxn
        [d,D] = size(X);
        J = (1/D) * sum(diag(l),'all') + lambda * (sum(W{1}.^2,'all')+sum(W{2}.^2,'all'));

end