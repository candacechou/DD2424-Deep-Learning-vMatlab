% function acc = ComputeAccuracy(X, y, W, b)
% computes the accuracy of the network’s predictions.the accuracy of a classifier for a given set of examples is the percentage of
% examples for which it gets the correct answer.
%     input:
%         X = dxn
%         y = 1xn 
%         W = cell(n,4)
%         b = cell(n,4)
%     output:
%         acc = 1x1  scalar value containing the accuracy.

function acc = ComputeAccuracyEnsemble(X, y, W, b)
[d,n] = size(X);   
[K,d] = size(W{1,2});
accumulater = zeros(K,n);
    for i=1:size(W,1)
        [h,P] = EvaluateClassifier(X,W(i,:),b(i,:));
        [~,ii] = max(P);
        ii = reshape(ii,size(ii,2),1);
        
        for j=1:K
            a = find(ii==j);
            accumulater(j,a) = accumulater(j,a) + i;
        end
    end
   [~, test_result] = max(accumulater);
   test_result = reshape(test_result,size(test_result,2),1);
   acc = size(find(test_result==y),1) / size(test_result,1);
    
end