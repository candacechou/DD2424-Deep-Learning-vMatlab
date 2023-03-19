% function acc = ComputeAccuracy(X, y,net,k_layers)
% computes the accuracy of the network’s predictions.the accuracy of a classifier for a given set of examples is the percentage of
% examples for which it gets the correct answer.
%     input:
%         X         = dxn
%         y         = 1xn 
%         net       = net
%         k_layers  = 1x1
%         t         = 1x1 for evaluate classifier
%         BN        = 1x1
%     output:
%         acc       = 1x1  scalar value containing the accuracy.

function acc = ComputeAccuracy(X, y,net,k_layers,t,P)
%     if BN == 0
%         [P,h,s] = EvaluateClassifier(X,net,k_layers);
%     else
%         [P,Xhat_batch,s,s_hat,net] = EvaluateClassifier_BN(X,net,k_layers,t,1);
%     end
    [~,ii] = max(P);
    ii = reshape(ii,size(ii,2),1);
    acc = size(find(ii==y),1) / size(ii,1);
    
end