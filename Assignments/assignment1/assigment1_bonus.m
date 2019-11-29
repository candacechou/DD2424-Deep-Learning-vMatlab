%%%%% this file is the SVM multi-class loss function

s                     = rng(400);
[X, Y, y]             = LoadBatch('data_batch_1.mat');
split                 = 7000;
trainX                = X(:,1:split);
trainY                = Y(:,1:split);
trainy                = y(1:split);
valX                  = X(:,split:end);
valY                  = Y(:,split:end);
valy                  = y(:,split:end);
[testX, testY, testy] = LoadBatch('test_batch.mat');
d                     = 32 * 32 * 3;
N                     = 10000;
K                     = 10;
rng(s);
%%% Xavier
x                     = 1/sqrt(32*32*3);
W                     = x * randn(K,d);
b                     = x * randn(K,1);
lambda                = 0.01;
n_batch               = 100;
eta                   = 0.01;
n_epochs              = 4;
C                     = 1;
training_loss         = zeros(n_epochs,1);
validation_loss       = zeros(n_epochs,1);
for i=1:n_epochs
    [W,b]               = MiniBatchSVM(trainX,trainy,trainY,n_batch,eta,n_epochs,W,b,lambda);
    t_cost              = ComputeCostSVM(trainX,trainY,W,b,lambda,C);
    training_loss(i,1)  = t_cost;
    v_cost              = ComputeCostSVM(valX,valY,W,b,lambda,C);
    validation_loss(i,1)= v_cost;
    if rem(i,10)==0
        eta                 = eta * 0.9;
    end
end
yline = 1:n_epochs;
figure()
a1      = plot(yline,training_loss,'g-');
M1      = 'Training loss ';
hold on
a2      = plot(yline,validation_loss,'b-');
M2      = 'Validation loss ';
legend(M1,M2);
title(" Cost each Epoch");
xlabel(" Epoch ");
ylabel(" Cost Function ");
hold off

figure()
for i=1:10
    im      = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
    subplot(2,5,i);
    imshow(s_im{i});
    
end
ComputeAccuracySVM(testX,testy,W,b,testY)
function J = ComputeCostSVM(X, Y, W, b, lambda,C)  
        J       = 0;
        
        P       = EvaluateClassifierSVM(X,W,b,Y);
        [d,D]   = size(X);
        %J = C * sum(sum(P)) + lambda * sum(W.^2,'all');
        J       = 1/D * (sum(sum(P)-1)) + lambda * sum(W.^2,'all');
        
end
function [grad_w, grad_b] = ComputeGradientsSVM(X,y,Y,b,W,lambda)
    [K,n]       = size(X);
    [K,d]       = size(W);
    grad_w      = zeros(K,d);
    grad_b      = zeros(K,1);

    P           = EvaluateClassifierSVM(X,W,b,Y);
    for i=1:n
        xi      = X(:,i);
        Pi      = P(:,i);
        gi      = repmat(xi',K,1);
        gb      = zeros(K,1);
        a       = find(Pi > 0 & Pi~=1);
        gb(a)   = 1;
        a       = find(Pi == 1);
        gi(a,:) = -gi(a,:)*(length(find(Pi > 0&Pi~=1)));
        gb(a)   = -length(find(Pi>0&Pi~=1));
        a       = find(Pi == 0);
        gi(a,:) = 0;
        gb(a)   = 0;
       
        grad_w  = grad_w + gi;
        grad_b  = grad_b + gb;
    end
 grad_w         = grad_w./n + 2*lambda*W;
 grad_b         = grad_b./n;
end
    
function acc = ComputeAccuracySVM(X, y, W, b,Y)
    P           = EvaluateClassifierSVM(X,W,b,Y);
    [~,ii]      = max(P);
    ii          = reshape(ii,size(ii,2),1);
    acc         = size(find(ii==y),1) / size(ii,1);
    
end
function P = EvaluateClassifierSVM(X,W,b,Y)
    [d,n]       = size(X);
    [k,d]       = size(W);
    P           = zeros(k,n);
    bm          = repmat(b,1,n);
    temp        = W * double(X) + bm;
    temp_w      = Y .* temp;
    c           = find(temp_w ~=0);
    xd          = temp(c);
    temp_x      = repmat(xd',k,1);
    ss          = temp - temp_x + 1;
        
    P           = max(0,ss);
end
function [Wstar, bstar] = MiniBatchSVM(X, y,Y, n_batch,eta,n_epochs, W, b, lambda)
    [d,n]       = size(X);
    Wstar       = W;
    bstar       = b;
%   s           = rng(400);
%   rng(s);
%   rand_i      = randperm(n);
    rand_i      = 1:n;

    shuffle_x   = X(:,rand_i);
    shuffle_y   = y(rand_i);
    shuffle_Y   = Y(:,rand_i);
    for j=1:n/n_batch
        j_start             = (j-1) * n_batch + 1;
        j_end               = j * n_batch;
        inds                = j_start:j_end;
        Xbatch              = shuffle_x(:, j_start:j_end);
        Ybatch              = shuffle_Y(:, j_start:j_end);
        ybatch              = shuffle_y(j_start:j_end,:);
        [grad_W, grad_b]    = ComputeGradientsSVM(Xbatch, ybatch,Ybatch,bstar,Wstar,lambda);
        %[grad_b, grad_W]   = ComputeGradsNum(Xbatch, Ybatch, Wstar, b, lambda, 1e-6);
        Wstar               = Wstar - eta * grad_W;
        bstar               = bstar - eta * grad_b;
    % for stochastic
   
    
 end
end
function [X, Y, y] = LoadBatch(filename)


    d       = 32 * 32 * 3;
    N       = 10000;
    K       = 10;
    X       = zeros(d,N);
    Y       = zeros(K,N);
    y       = zeros(1,N);
 
    A       = load(filename);
    I       = reshape(A.data',32,32,3,10000);
    X       = reshape(I,32*32*3,10000);
    X       = double(X)./255;
    labels  = A.labels;
    %%% one-hot representation for Y
    for i = 1:10 
        row         = find(labels == i-1);
        Y(i,row)    = 1;
    end
    y       = labels + 1;
    
   
end
