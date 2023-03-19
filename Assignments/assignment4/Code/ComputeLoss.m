function loss = ComputeLoss(X_char,Y_char,RNN,h)
[K,n]   = size(X_char);
loss      = 0;
for t =1 : n
    a_temp = RNN.W * h + RNN.U * X_char(:, t) + RNN.b;
    h = tanh(a_temp);
    o = RNN.V * h + RNN.c;
    p_temp = exp(o);
    p = p_temp/sum(p_temp);

    loss = loss - log(Y_char(:, t)' * p);
end

end