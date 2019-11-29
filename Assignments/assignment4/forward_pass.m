% function [loss,a,p,h,o]= forward_pass(RNN,X_chars,Y_chars,h0,n,book_char)
%     Input : 
%         RNN         = parameter structure
%         X_char      = first sequence characters of the book
%         Y_char      = labelled sequence for debugging
%         h0          =  ht
%         n           = length of sequence
%         book_char   = for the one hot representation
%     Output:
%         loss        = calculated loss
%         a,p,h,o     = is going to be used in backward_pass
%     
function [loss, a, p, h, o] = forward_pass(RNN,X_char,Y_char,h0,n,book_char,m)
[~,K]   = size(book_char);
o       = zeros(K, n);
p       = zeros(K, n);
h       = zeros(m, n);
a       = zeros(m, n);
h_t  = h0;
loss    = 0;
for t =1 : n
    a_t = RNN.W * h_t + RNN.U * X_char(:, t) + RNN.b;
    a(:, t) = a_t;
    h_t = tanh(a_t);
    h(:, t) = h_t;
    o(:, t) = RNN.V * h_t + RNN.c;
    p_t = exp(o(:, t));
    p(:, t) = p_t/sum(p_t);

    loss = loss - log(Y_char(:, t)' * p(:, t));
end
% loss = real(loss);
h = [h0, h];

end
