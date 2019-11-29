rng shuffle
%%%% Read in the data
book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);
book_char = unique(book_data);
Key_set = num2cell(book_char);
fin = fopen('result.txt','w');
%%%% create map containers
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');
value_set = 1:length(Key_set);
char_to_ind = containers.Map(Key_set,value_set);
ind_to_char = containers.Map(value_set,Key_set);
book_Y      = OneHot(book_data,book_char);
%%%% Initialize hyper-parmeters
[~,n]       = size(book_data);
[~,k]       = size(book_char);
m           = 100;
sig         = 0.1;
eta         = 0.1;
RNN.b       = zeros(m,1);
RNN.c       = zeros(k,1);
RNN.U       = randn(m,k)*sig;
RNN.W       = randn(m,m)*sig;
RNN.V       = randn(k,m)*sig;

mRNN.b       = zeros(m,1);
mRNN.c       = zeros(k,1);
mRNN.U       = zeros(m,k);
mRNN.W       = zeros(m,m);
mRNN.V       = zeros(k,m);

best_RNN     = RNN;
best_loss    = 300;
best_y       = [];
%%%% Check gradient
X_ind = zeros(1, length(book_data));
% for i = 1 : length(book_data)        
%     X_ind(i) = char_to_ind(book_data(i));
% end
% batch_size = 15;
% 
seq_length = 20;
% X_char = book_Y(:,1:seq_length);
% Y_char = book_Y(:,2:seq_length+1);
% 
% Checking_Gradient(X_char,Y_char, RNN, k,seq_length,book_char,m);
book_X = book_Y;     
e = 1;   %%%%
textlen = 1000;
n_epochs = 10;
Loss = [];
loss = 0;
hprev = [];

iter = 1;
% seq_length_list = [40,40,40,30,30,30,25,25,25,20,20,20,15,15,15,10,10,10,5,5,5];
for i=1:n_epochs
disp(['new epoch:' num2str(i),'\n']);
% seq_length = seq_length_list(i);
   while e <=length(book_data) - seq_length - 1
    X_batch = book_X(:, e : e + seq_length - 1);
    Y_batch = book_Y(:, e + 1 : e + seq_length);
    
    if e == 1
        hprev = zeros(m, 1);
        
    else
        hprev = h(:, end);
    end
    [loss, a, p, h, o] = forward_pass(RNN,X_batch,Y_batch,hprev,seq_length,book_char,m);
    grads              = ComputeGradient(RNN, X_batch, Y_batch, a, h, p, seq_length,m);
    for f = fieldnames(RNN)'
        %%%% clip the gradient 
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
        %%%% AdaGrad
        mRNN.(f{1}) = mRNN.(f{1}) + grads.(f{1}).^2;
        RNN.(f{1}) = RNN.(f{1}) - eta * (grads.(f{1})./(mRNN.(f{1}) + eps).^(0.5));
       %%%% vanilla
%        RNN.(f{1}) = RNN.(f{1}) - eta * grads.(f{1});
     
    end
    
    if iter == 1 && e == 1
        smooth_loss = loss;
    else
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss;
    end
    if smooth_loss < best_loss 
        best_loss = smooth_loss;
        best_RNN  = RNN;
        best_y = Syn_text(RNN,hprev,X_batch(:, 1),textlen,book_char);
    end
    Loss = [Loss,smooth_loss];
    
    if iter == 1 || mod(iter, 1000) == 0
        y = Syn_text(RNN,hprev,X_batch(:, 1),textlen,book_char);
        c = [];
%         loss
        for j = 1 : textlen
            c = [c ind_to_char(y(j))];
        end
        fprintf(fin,'\n-----------------------------------------------------------\n');
        fprintf(fin,'iter = ')
        fprintf(fin, num2str(iter))
        fprintf(fin, ', smooth_loss = ')
        fprintf(fin, num2str(smooth_loss))
        fprintf(fin,'\n');
        fprintf(fin,c);
    end
    e    = e + seq_length;
    iter = iter + 1
   end
        
        e  = 1;
   end

   %%%% print the best result
   disp(['best results:'])
   for j = 1 : textlen
        c = [c ind_to_char(y(j))];
   end
  fprintf(fin,'\n-----------------------------------------------------------');
  fprintf(fin, ' smallest smooth_loss = ', num2str(best_loss));
  c = [];
  for j = 1 : textlen
            c = [c ind_to_char(best_y(j))];
        end
  fprintf(fin,c);
  fclose(fin);
  figure()
  plot(1:length(Loss),Loss)
  M1 = 'Loss ';
  legend(M1);
  title(" Loss each update step");
  xlabel(" update step  ");
  ylabel("Loss");
  hold off
function Y = OneHot(book_data,book_char)
    [~,n] = size(book_data);
    [~,k] = size(book_char);
    Y = zeros(k,n);
    for i=1:k
        [~,ii] = find(book_data == book_char(i));
        Y(i,ii) = 1;
    end
end
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
function num_grads = ComputeGradsNum(X, Y, RNN, h)

for f = fieldnames(RNN)'
%     disp('Computing numerical gradient for')
%     disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = ComputeLoss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = ComputeLoss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end
end

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
  