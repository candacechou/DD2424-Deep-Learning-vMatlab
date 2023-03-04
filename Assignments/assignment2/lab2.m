clc,clear all

[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[ValX, ValY, Valy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');
[Flip_X,Flip_Y,Flip_y] = flipdata('data_batch_1.mat');
% TRAIN WITH MORE DATA
% [trainX1, trainY1, trainy1] = LoadBatch('data_batch_1.mat');
% [trainX2, trainY2, trainy2] = LoadBatch('data_batch_2.mat');
% [trainX3, trainY3, trainy3] = LoadBatch('data_batch_3.mat');
% [trainX4, trainY4, trainy4] = LoadBatch('data_batch_4.mat');
% [trainX5, trainY5, trainy5] = LoadBatch('data_batch_5.mat');
% [testX, testY, testy] = LoadBatch('test_batch.mat');
% trainX = [trainX1,trainX2,trainX3,trainX4,trainX5(:,1:9000)];
% trainY = [trainY1,trainY2,trainY3,trainY4,trainY5(:,1:9000)];
% trainy = [trainy1;trainy2;trainy3;trainy4;trainy5(1:9000)];
% ValX = trainX5(:,9001:end);
% ValY = trainY5(:,9001:end);
% Valy = trainy5(9001:end);
[d,n] = size(trainX);
f = 10;
l_min = -3;
l_max = -1;
k_num = 10;
m = 50;
std_1 = 1/sqrt(d);
std_2 = 1/sqrt(m);
mean_1 = 0 ;
mean_2 = 0;
[W,b, lambda]= InitializeParameter(trainX,k_num,std_1,mean_1,std_2,mean_2,l_min,l_max,m);


%%%% testing the gradient 
% [h,P] = EvaluateClassifier(trainX, W,b);
% [grad_W,grad_b] = ComputeGradients(trainX, trainY, P,h,W,b,lambda);
% [grad_b_test, grad_W_test] = ComputeGradsNumSlow(trainX, trainY, W, b, lambda, 1e-5);
%%% testing the gradient function
%%% - testing -> absolute only
% W1_a = abs(grad_W{1} - grad_W_test{1});
% find(W1_a > 1e-6)
% W2_a = abs(grad_W{2} - grad_W_test{2});
% find(W2_a > 1e-6)
% b1_a = abs(grad_b{1} - grad_b_test{1});
% find(b1_a > 1e-6)
% b2_a = abs(grad_b{2} - grad_b_test{2});
% find(b2_a > 1e-6)
% %%%% - testing -> with more accurate 
% h = 1e-6;
% W1_a = abs(grad_W{1} - grad_W_test{1})./max(h,abs(grad_W{1})+abs(grad_W_test{1}));
% find(W1_a > 1e-4)
% W2_a = abs(grad_W{2} - grad_W_test{2})./max(h,abs(grad_W{2})+abs(grad_W_test{2}));
% find(W2_a > 1e-4)
% b1_a = abs(grad_b{1} - grad_b_test{1})./max(h,abs(grad_b{1})+abs(grad_b_test{1}));
% find(b1_a > 1e-4)
% b2_a = abs(grad_b{2} - grad_b_test{2})./max(h,abs(grad_b{2})+abs(grad_b_test{2}));
% find(b2_a > 1e-4)

eta_min = 1e-5;
eta_max = 1e-1;

l = 0;
t = 1;
n_epoch = 10;
n_batch = 100;
ns = 500; %2*floor(n/ n_batch);
v = 11;
t_loss = zeros(v,1);
v_loss = zeros(v,1);
t_cost = zeros(v,1);
v_cost = zeros(v,1);
t_acc = zeros(v,1);
v_acc = zeros(v,1);
etas = zeros(1000,1);
% ts = zeros(n_epochs,1);
% ls = zeros(n_epochs,1);
%%%% emsemble 
w_new = cell(4,2);
b_new = cell(4,2);
i = 2;
yline = zeros(v,1);
t_cost(1,1) = ComputeCost(trainX,trainY,W,b,lambda);
t_loss(1,1) = ComputeLoss(trainX,trainY,trainy,W,b);
t_acc(1,1) = ComputeAccuracy(trainX,trainy,W,b);
v_cost(1,1) = ComputeCost(ValX,ValY,W,b,lambda);
v_loss(1,1) = ComputeLoss(ValX,ValY,Valy,W,b);
v_acc(1,1) = ComputeAccuracy(ValX,Valy,W,b);
for k=1:n_epoch    
rand_i = randperm(n);

FlipX_shuffle = Flip_X(:,rand_i);
FlipY_shuffle = Flip_Y(:,rand_i);
shuffle_x = trainX(:,rand_i);
shuffle_y = trainY(:,rand_i);
FlipX_shuffle = Flip_X(:,rand_i);
FlipY_shuffle = Flip_Y(:,rand_i);
for j=1:n/n_batch
    j_start = (j-1) * n_batch + 1;
    j_end = j * n_batch;
    flip_start = (j-1)*f + 1;
    flip_end = (j-1)*f + 1 + f;
    inds = j_start:j_end;
    Xbatch = shuffle_x(:, j_start:j_end);
    Ybatch = shuffle_y(:, j_start:j_end);
    
    Xbatch = [shuffle_x,FlipX_shuffle(:,flip_start:flip_end)];
    Ybatch = [shuffle_y,FlipY_shuffle(:,flip_start:flip_end)];

    [h,P] = EvaluateClassifier(Xbatch, W, b);
    [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch,P,h,W,b,lambda);
    if (t >= 2 * l * ns) && (t < (2*l + 1) * ns)
        etas_unit = eta_min + ((t - 2 * l * ns) / ns) * (eta_max - eta_min);
    elseif (t>=(2*l + 1) * ns) && t <(2*(l+1)*ns)
        etas_unit = eta_max - ((t - (2 * l+1) * ns) / ns) * (eta_max - eta_min);
    end
    
    etas(t,1) = etas_unit;



    W{1} = W{1} - etas_unit*grad_W{1};
    b{1} = b{1} - etas_unit*grad_b{1};
    W{2} = W{2} - etas_unit*grad_W{2};
    b{2} = b{2} - etas_unit*grad_b{2};
    t = t + 1;
    
    if rem(t,2*ns) == 0
        l = l+1;     
        
    end
    if etas_unit == eta_min
        w_new(l,:) = W;
        b_new(l,:) = b;
    end
    if rem(t,100) == 0
        t_cost(i,1) = ComputeCost(trainX,trainY,W,b,lambda);
        t_loss(i,1) = ComputeLoss(trainX,trainY,trainy,W,b);
        t_acc(i,1) = ComputeAccuracy(trainX,trainy,W,b);
        v_cost(i,1) = ComputeCost(ValX,ValY,W,b,lambda);
        v_loss(i,1) = ComputeLoss(ValX,ValY,Valy,W,b);
        v_acc(i,1) = ComputeAccuracy(ValX,Valy,W,b);
        yline(i,1) = t;
        i = i+1;
    end
   t
end   
%     etas(i,1) = eta;
%     ts(i,1) = t;
%     ls(i,1) = l;
    
    
end
%acc = ComputeAccuracyEnsemble(testX,testy,w_new,b_new)
accno = ComputeAccuracy(testX,testy,W,b)
figure()
a1 = plot(yline,t_loss,'g-');
M1 = 'Training loss ';
hold on
a2 = plot(yline,v_loss,'b-');
M2 = 'Validation loss ';
legend(M1,M2);
title(" Loss each update step");
xlabel(" update step  ");
ylabel(" Loss Function ");
% ylim([0,2.5]);
hold off

figure()
a1 = plot(yline,t_cost,'g-');
M1 = 'Training cost ';
hold on
a2 = plot(yline,v_cost,'b-');
M2 = 'Validation cost ';
legend(M1,M2);
title(" Cost each update step");
xlabel(" update step  ");
ylabel(" Cost Function ");
% ylim([0,3]);
hold off

figure()
a1 = plot(yline,t_acc,'g-');
M1 = 'Training Accuracy ';
hold on
a2 = plot(yline,v_acc,'b-');
M2 = 'Validation Accuracy ';
legend(M1,M2);
title(" Accuracy each update step");
xlabel(" update step  ");
ylabel(" Accuracy ");
% ylim([0,0.7]);
hold off

figure()
a1 = plot(etas,'g-');
M1 = 'etas ';
legend(M1);
title(" etas");
xlabel(" update step  ");
ylabel("etas ");
hold off

function [W,b, lambda] = InitializeParameter(X,k_num,std_1,mean_1,std_2,mean_2,l_min,l_max,m)
[d,n] = size(X);
%m = 1/std_2^2;
l = l_min + (l_max - l_min)*rand(1, 1); 
lambda = 10^l;
W1 = zeros(m,d);
W1 = std_1 * randn(m,d) + mean_1;
b1 = zeros(m,1);
W2 = zeros(k_num,m); %%%% not sure
W2 = std_2 * randn(k_num,m') + mean_2;
b2 = zeros(k_num,1);
W = cell(1,2);
W = {W1,W2};
b = {b1,b2};
end

function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    I = reshape(A.data',32,32,3,10000);
    X = reshape(I,32*32*3,10000);
    X = double(X);
    X = X./255;
    mean_X = mean(X,2);
    std_X = std(X,0,2);
    X = X - repmat(mean_X, [1, size(X, 2)]); 
    X = X ./repmat(std_X, [1, size(X, 2)]);
    labels = A.labels;
    %%% one-hot representation for Y
    for i = 1:10 
        row = find(labels == i-1);
        Y(i,row) = 1;
    end
    y = labels + 1;
 
end

function acc = ComputeAccuracy(X, y, W, b)
    [h,P] = EvaluateClassifier(X,W,b);
    [~,ii] = max(P);
    ii = reshape(ii,size(ii,2),1);
    acc = size(find(ii==y),1) / size(ii,1);
    
end

function J = ComputeCost(X, Y, W, b, lambda) 
        J = 0;
        [h,P] = EvaluateClassifier(X,W,b);
        l = -log(Y'* P); %% nxn
        [d,D] = size(X);
        J = (1/D) * sum(diag(l),'all') + lambda * (sum(W{1}.^2,'all')+sum(W{2}.^2,'all'));

end

function loss = ComputeLoss(X,Y,y,W,b)
loss = 0;
[h,P] = EvaluateClassifier(X,W,b);
loss_matrix = -log(Y'* P);
[d,D] = size(X);
loss = sum(diag(loss_matrix),'all')/D;




end

function [h,P] = EvaluateClassifier(X, W,b)
        [d,n] = size(X);
        [m,d] = size(W{1});
        k=10;
        P = zeros(k,n);
        b1_m  = repmat(b{1},1,n);
        %%%% The first layer
        s1 = W{1} * X + b1_m;
        h = max(0,s1);  %%%% kxn
        b2_m = repmat(b{2},1,n);
        %%%% The second layer
        P = W{2} * h + b2_m;
        %%%% softmax
        P = exp(P);
        Psum = sum(P,1);
        Psum = repmat(Psum,k,1);
        P = P ./Psum; 

       
        
end

function [X,Y,y] = flipdata(filename)
    A = load(filename);
    I = reshape(A.data',32,32,3,10000);
    I = flip(I);
    X = reshape(I,32*32*3,10000);
    X = double(X);
    X = X./255;
    mean_X = mean(X,2);
    std_X = std(X,0,2);
    X = X - repmat(mean_X, [1, size(X, 2)]); 
    X = X ./repmat(std_X, [1, size(X, 2)]);
    labels = A.labels;
    %%% one-hot representation for Y
    for i = 1:10 
        row = find(labels == i-1);
        Y(i,row) = 1;
    end
    y = labels + 1;
  
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P,h, W,b, lambda)
    [K,n] = size(Y);
    [d,n] = size(X);
    grad_W = cell(1,2);
    grad_b = cell(1,2);

    %%%% gradient of W2 and b2
    g = - (Y - P);
    grad_W{2} = g * h'./n + 2 * lambda * W{2} ;
    grad_b{2} = sum(g,2)./n;
    %%%% gradient of W1 and b1
    g_batch = W{2}'* g; %%% kxn
    ind_h = h > 0; %%%% kxn
    g_batch = g_batch .* ind_h; %%%% kxn
    grad_W{1} = g_batch * X'./n + 2 * lambda * W{1};
    grad_b{1} = sum(g_batch,2)./n;
   
end
