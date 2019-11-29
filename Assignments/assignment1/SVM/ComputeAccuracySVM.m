% function acc = ComputeAccuracySVM(X, y, W, b)
% computes the accuracy of the network’s predictions.the accuracy of a classifier for a given set of examples is the percentage of
% examples for which it gets the correct answer.
%     input:
%         X = dxn
%         y = 1xn 
%         W = Kxd
%         b = Kx1
%         Y = Kxn
%     output:
%         acc = 1x1  scalar value containing the accuracy.

function acc = ComputeAccuracySVM(X, y, W, b,Y)
    P       = EvaluateClassifierSVM(X,W,b,Y);
    [~,ii]  = max(P);
    ii      = reshape(ii,size(ii,2),1);
    acc     = size(find(ii==y),1) / size(ii,1);
    
end