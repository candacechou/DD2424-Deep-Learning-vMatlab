% function y = Syn_text(RNN,h0,x0,Y,n)
%     Input:
%         RNN = 1x1 The data structure for RNN
%         h0  = The hidden state vector at time 0 
%         x0  = The first dummy input vector to the RNN
%         Y   = One hot representation
%         n   = the length of the sequence we want to generate.
%         book_char = for one hot function
%     Output:
%         y = output one hot kxn
%         
     
function y = Syn_text(RNN,h0,x0,n,book_char)
h = h0;
x = x0;
y = zeros(1, n);
for i=1:n
    a   = RNN.W * h + RNN.U * x + RNN.b;
    h   = tanh(a);
    o   = RNN.V * h + RNN.c;
    p   = exp(o);
    p   = p/sum(p);
%%%% randomly select a character based on the output prob

    cp   = cumsum(p);
    a    = rand;
    ixs  = find(cp-a>0);
    ii   = ixs(1);
    x    = OneHot(ii,book_char);
    y(i) = ii;
end
end
    