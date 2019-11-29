% function grad = ComputeGradient(RNN, X_chars, Y_chars, a, h, p, n)
% 
%     Input:
%       RNN         = parameter structure
%       X_char      = first sequence characters of the book
%       Y_char      = labelled sequence for debugging
%       [a,h,p]     = Compute in backward_pass
%       n           = length of sequence
%     Output:
%       grad        = gradient structure
%                   = grad.W,grad.U,grad.V,grad.b,grad.c
function grad = ComputeGradient(RNN, X_chars, Y_chars, a, h, p, n,m)
    g_h = zeros(n, m);
    g_a = zeros(n, m);
    g = -(Y_chars - p)';
    grad.c = sum(g)';
    grad.V = g'*h(:,2:end)';  
    g_h(n, :) = g(n, :) * RNN.V;                                  
    g_a(n, :) = g_h(n, :) * diag(1 - (tanh(a(:, n))).^2);       
    for t = n-1:-1:1
        g_h(t, :) = g(t, :) * RNN.V + g_a(t + 1, :)*RNN.W;
        g_a(t, :) = g_h(t, :) * diag(1 - (tanh(a(:, t))).^2);
    end

grad.b = (sum(g_a))';                                         
grad.W = g_a' * h(:, 1 : end - 1)';                        
grad.U = g_a' * X_chars';                                      

end