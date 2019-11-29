% function [G_batch] = BatchNormBackPass(Gbatch,S_batch,mu,v)
%   do BatchNormBackPass
%   Input :
%       Gbatch         = k_layers{l}xn
%       S_batch        = k_layers{l}xn
%       mu             = 1x1
%       v              = dxn
%   
%    Output :
%       G_batch        = k_layers{l}xn

function [G_batch] = BatchNormBackPass(Gbatch,S_batch,mu,v) 
   [col,n]             = size(Gbatch);
   sigma_1             = ((v+eps).^(-0.5));
   sigma_2             = ((v+eps).^(-1.5));
   I_n                 = ones(1,n); %%%%% In' in the instruction
   G1                  = Gbatch .* (sigma_1 * I_n);
   G2                  = Gbatch .* (sigma_2 * I_n);
   D                   = S_batch - mu * I_n;
   c                   = (G2 .* D) * I_n';
   G_batch             = G1 - mean(G1,2) - D.*(c * I_n)/n;
  
end
