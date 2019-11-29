%%%%% This file is to test the ensemble method on bonus in assignment 1
s                           = rng(400);
[trainX, trainY, trainy]    = LoadBatch('data_batch_1.mat');
[valX, valY, valy]          = LoadBatch('data_batch_2.mat');
[testX, testY, testy]       = LoadBatch('test_batch.mat');
%%%%% 2
% [d,Nt]                    = size(testX);
% [d,Ntr]                   = size(trainX);
% [d,Nv]                    = size(testX);
d                           = 32 * 32 * 3;
N                           = 10000;
K                           = 10;
rng(s);
n_batch                     = [100,200,50,400,300,20,15,10,5,1];
eta                         = [0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.035];
n_epochs                    = [130,400,40,300,80,100,150,200,50,250];
lambda                      = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1];
accu                        = zeros(K,000);
w_acc                       = zeros(10,K,d);
b_acc                       = zeros(K,10);
n_ens                       = randi([1,10],1);
for ens=1:20
    testing                 = randi([1,3],1);
    x                       = sqrt(1/3072);
    W                       = x*ens^2 * randn(K,d);
    b                       = x*ens^2 * randn(K,1);
    i_lambda                = randi([1,11],1);
    i_batch                 = randi([1,10],1);
    i_ep                    = randi([1,10],1);
    i_eta                   = randi([1,10],1);
    batc                    = n_batch(i_batch);
    lamb                    = lambda(i_lambda);
    etas                    = eta(i_eta);
    epoch                   = n_epochs(i_ep);
    
    
    if testing == 1
        for i=1:epoch
        [W,b]               = MiniBatchGD(trainX,trainY,batc,etas,epoch,W,b,lamb);
        w_acc(ens,:,:)      = W ;
        b_acc(:,ens)        = b ;
    end
    elseif testing == 2
            for i=1:epoch
                [W,b]       = MiniBatchGD(trainX,trainY,batc,etas,epoch,W,b,lamb);
                if rem(i,10)==0
                    etas    = etas / 10;
                end
                w_acc(ens,:,:)      = W ;
                b_acc(:,ens)        = b ;
            end
        else
            for i=1:epoch
                [W,b]       = MiniBatchGD(trainX,trainY,batc,etas,epoch,W,b,lamb);
                etas        = etas * 0.9;
            end
            w_acc(ens,:,:)      = W ;
            b_acc(:,ens)        = b ;
    end
    
    ens
    end


acc = ComputeAccuracyEnsemble(testX,testy,w_acc,b_acc)

function [X, Y, y] = LoadBatch(filename)


    d = 32 * 32 * 3;
    N = 10000;
    K = 10;
    X = zeros(d,N);
    Y = zeros(K,N);
    y = zeros(1,N);
 
    A = load(filename);
    I = reshape(A.data',32,32,3,10000);
    X = reshape(I,32*32*3,10000);
    X = double(X)./255;
    labels = A.labels;
    %%% one-hot representation for Y
    for i = 1:10 
        row = find(labels == i-1);
        Y(i,row) = 1;
    end
    y = labels + 1;
    
   
end
function J = ComputeCost(X, Y, W, b, lambda)  
        J = 0;
        P = EvaluateClassifier(X,W,b);
        l = -log(Y'* P); %% nxn
        [d,D] = size(X);
        J = (1/D) * sum(diag(l),'all') + lambda * sum(W.^2,'all');
        
end
function acc = ComputeAccuracyEnsemble(X, y, W, b)
    [a,K,d] = size(W);
    [d,n] = size(X);
    accumulater = zeros(K,n);
    for i=1:a
        temp_w = reshape(W(i,:,:),[K,d]);
        
        P = EvaluateClassifier(X,temp_w,b(:,i));
        [~,ii] = max(P);
        for j=1:K
            a = find(ii==j);
            accumulater(j,a) = accumulater(j,a)+1;
        end
    end
   [~, test_result] = max(accumulater);
   test_result = reshape(test_result,size(test_result,2),1);
    acc = size(find(test_result==y),1) / size(test_result,1);
   
end
function [grad_w, grad_b] = ComputeGradients(X,Y,P,W,lambda)
    [K,n] = size(Y);
    [K,d] = size(W);
    grad_w = zeros(K,d);
    grad_b = zeros(K,1);
    g = -(Y-P)';
    X = double(X);
    grad_b = sum(g,1)'./n;  
    grad_w = g' * X'./n + 2 * lambda * W;   
end
function P = EvaluateClassifier(X,W,b)
        X = double(X);
        [d,n] = size(X);
        [k,d] = size(W);
        P = zeros(k,n);
        bm = repmat(b,1,n);
        P = W * double(X) + bm;
        P = exp(P);
        Psum = sum(P,1);
        Psum = repmat(Psum,k,1);
        P = P ./Psum;       
end
function [Wstar, bstar] = MiniBatchGD(X, Y, n_batch,eta,n_epochs, W, b, lambda)
[K,n] = size(Y);
[d,n] = size(X);
Wstar = W;
bstar = b;
% s = rng(400);
% rng(s);
%rand_i = randperm(n);
rand_i = 1:n;

shuffle_x = X(:,rand_i);
shuffle_y = Y(:,rand_i);
for j=1:n/n_batch
    j_start = (j-1) * n_batch + 1;
    j_end = j * n_batch;
    inds = j_start:j_end;   
    Xbatch = shuffle_x(:, j_start:j_end);
    Ybatch = shuffle_y(:, j_start:j_end);
    P = EvaluateClassifier(Xbatch, Wstar, bstar);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch,P,Wstar,lambda);
    %[grad_b, grad_W] = ComputeGradsNum(Xbatch, Ybatch, Wstar, b, lambda, 1e-6);
    Wstar = Wstar - eta * grad_W;
    bstar = bstar - eta * grad_b;
    % for stochastic
end
end