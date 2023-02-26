%%%%% % function acc = ComputeAccuracyEnsemble(X, y, W, b)
% computes the accuracy of the network’s predictions. The accuracy of a classifier for a given set of examples is the percentage of
% examples for which it gets the correct answer.
%     input:
%         X     = d x n
%         y     = 1 x n 
%         W     = a x K x d
%         b     = K x a
%     output:
%         acc   = 1x1  scalar value containing the accuracy.

function acc = ComputeAccuracyEnsemble(X, y, W, b)
    [a,K,d]     = size(W);
    [d,n]       = size(X);
    accumulater = zeros(K,n);
    for i=1:a
        temp_w      = reshape(W(i,:,:),[K,d]);
        
        P           = EvaluateClassifier(X,temp_w,b(:,i));
        [~,ii]      = max(P);
        for j=1:K
            a                   = find(ii==j);
            accumulater(j,a)    = accumulater(j,a)+1;
        end
    end
   [~, test_result]         = max(accumulater);
   test_result              = reshape(test_result,size(test_result,2),1);
   acc                      = size(find(test_result==y),1) / size(test_result,1);
   
end
        
        
            
