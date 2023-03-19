% function [J,loss] = ComputeCost(X, Y,net, lambda,k_layers,t)      
%  computes the cost function given by equation  for a set of images.
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
%         loss      =  1x1
function [J,loss] = ComputeCost(X, Y, net, lambda,k_layers,P,matrix) 

            J         = 0;
            [d,D]     = size(X);
            if matrix == 0
            l         = -log(Y'*P); %% nxn
            
%             J         = (1/D)*sum(l,'all');
             J         = (1/D) * sum(diag(l),'all');
            
            else 
                for j=1:D/1000
                    j_start           = (j-1) * 1000 + 1;
                    j_end             = j * 1000;    
                    inds              = j_start:j_end; 
                    Pbatch            = P(:,inds);
                    Ybatch            = Y(:,inds);
                    l = -log(Ybatch' * Pbatch);
                    J = J + trace(l);
                end
                J = J/D;
            end
            loss      = J;
        for i=1:k_layers
            J       = J + lambda* sum(sum(net.W{i,1}.^2));
        end

end