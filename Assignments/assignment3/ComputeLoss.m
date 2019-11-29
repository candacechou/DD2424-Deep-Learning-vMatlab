% function J = ComputeLoss(X, Y,net, lambda,k_layers)      
%  computes the loss function given by equation  for a set of images.
%  
%  input:
%         X         = dxn
%         Y         = Kxn
%         net       = net
%         lambda    = 1x1
%         k_layers  = 1x1
%  output:
%         J         = 1x1 scalar corresponding to the sum of the loss of the network’s predictions for the images in X relative to the 
%         ground truth labels and the regularization term on W.

function J = ComputeLoss(X, Y, net, lambda,k_layers) 
        J         = 0;
        if net.use_bn == 0
            [P,h,s]   = EvaluateClassifier(X,net,k_layers);
        else 
            
            [P,Xhat_batch,s,s_hat,net] = EvaluateClassifier_BN(X,net,k_layers);
        end
            l         = -log(Y'* P); %% nxn
            [d,D]     = size(X);
            J         = (1/D) * trace(l);
  
%         for i=1:k_layers
%             J       = J + lambda* sum(sum(net.W{i,1}.^2));
%         end

end