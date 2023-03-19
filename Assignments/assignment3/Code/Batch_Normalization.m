% function [S] = Batch_Normalization(Si,mu,v)
%     do batch_normalization on forward process
%     input :
%           Si                = dxn                each column of X corresponds to an image and it has size d×n.
%           mu                = dx1
%           v                 = dxn
%           
%     output :
%           S                 = dxn
function [S] = Batch_Normalization(Si,mu,v)
[d,n]       = size(Si);
S           = zeros(d,n);
S           = diag(v+eps)^(-0.5) * (Si - repmat(mu,1,n));
end