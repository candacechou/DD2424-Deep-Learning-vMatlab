function [] = CheckingGradient(net,grad_W,grad_b,k_layers,h)
% testing -> absolute only
% W1_a = abs(grad_W{1} - grad_W_test{1});
% find(W1_a > 1e-6)
% W2_a = abs(grad_W{2} - grad_W_test{2});
% find(W2_a > 1e-6)
% b1_a = abs(grad_b{1} - grad_b_test{1});
% find(b1_a > 1e-6)
% b2_a = abs(grad_b{2} - grad_b_test{2});
% find(b2_a > 1e-6)
    for i=1:k_layers
       fprintf('the %d th layers\n' ,i);
       Wa = abs(grad_W{i} - net.Grad_W{i,1});
       fprintf('W');
       find(Wa > h)
       Wb = abs(grad_b{i} - net.Grad_b{i,1});
       fprintf('b');
       find(Wb > h)
    end
       