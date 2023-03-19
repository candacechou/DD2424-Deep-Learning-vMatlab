function Checking_Gradient(X_char,Y_char, RNN, K,batch_size,book_char,m)
 
    eps = 1e-6;
    %% numerical computation
    n_grads = ComputeGradsNum(X_char(:, 1 : batch_size), Y_char(:, 1 : batch_size), RNN, eps);
    
    %% analytical gradients
    h0 = zeros(size(RNN.W, 1), 1);
    [loss, a, p, h, o] = forward_pass(RNN, X_char(:, 1 : batch_size), Y_char(:, 1 : batch_size),h0, batch_size, book_char,m);
    grads = ComputeGradient(RNN, X_char(:, 1 : batch_size), Y_char(:, 1 : batch_size),a, h, p, batch_size,m);
    
    
    
    for f = fieldnames(RNN)'
        
        n_g = n_grads.(f{1});
        a_g = grads.(f{1});
        den = abs(n_g) + abs(a_g);
        num = abs(n_g - a_g);
        fprintf('--------------------------------------\n');
        fprintf('Field name:')
        f{1}
        fprintf('\n rela error: \n');
        gradcheck_max = max(num(:))/max(eps, sum(den(:)))
        fprintf('absolute error max:\n')
        max(num(:))
    end

end