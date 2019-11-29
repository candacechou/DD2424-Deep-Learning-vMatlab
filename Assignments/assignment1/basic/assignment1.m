%clc, clear all ;
s                       = rng(400);
[X, Y, y]               = LoadBatch('data_batch_1.mat');
split                   = 7000;
trainX                  = X(:,1:split);
trainY                  = Y(:,1:split);
trainy                  = y(1:split);
valX                    = X(:,split:end);
valY                    = Y(:,split:end);
valy                    = y(:,split:end);
% [X1,Y1,y1]            = LoadBatch('data_batch_2.mat');
% [X2,Y2,y2]            = LoadBatch('data_batch_3.mat');
% [X3,Y3,y3]            = LoadBatch('data_batch_4.mat');
% [X4,Y4,y4]            = LoadBatch('data_batch_5.mat');
% split                 = 8500;
% trainX                = [X,X1,X2,X3,X4(:,1:split)];
% trainY                = [Y,Y1,Y2,Y3,Y4(:,1:split)];
% trainy                = [y;y1;y2;y3;y4(1:split)];
% valX                  = X4(:,split+1:end);
% valY                  = Y4(:,1+split:end);
% valy                  = y4(1+split:end);
%
%[valX, valY, valy]         = LoadBatch('data_batch_1.mat');
%[trainX, trainY, trainy]   = LoadBatch('data_batch_1.mat');
%[valX, valY, valy]         = LoadBatch('data_batch_1.mat');
[testX, testY, testy]       = LoadBatch('test_batch.mat');
%%%%% 2
d                   = 32 * 32 * 3;
N                   = 10000;
K                   = 10;
rng(s);
%%% Xavier
x                   = 1/sqrt(1);
W                   = 0.01 * randn(K,d);
b                   = 0.01* randn(K,1);
lambda              = 0.03;
h                   = 1e-6;
n_batch             = 10;
eta                 = 0.01;
n_epochs            = 40;
J                   = ComputeCost(trainX(:,1:2), trainY(:,1:2), W, b, lambda);
acc                 = ComputeAccuracy(trainX(:,1:200), trainy(1:200), W, b);
P                   = EvaluateClassifierSVM(trainX, W, b,trainY,trainy);
% % 
% [grad_W, grad_b]      = ComputeGradients(trainX, trainY,P,W,lambda);
% [grad_b1, grad_W1]    = ComputeGradsNum(trainX, trainY, W, b, lambda, h);
%  a                    = abs(grad_W - grad_W1)./max(h,abs(grad_W)+abs(grad_W1));
%  find(a > 1e-4)
%  b                    = abs(grad_b - grad_b1)./max(h,abs(grad_b)+abs(grad_b1));
% find (b >1e-4)
% a                     = abs(grad_W - grad_W1);
% find(a > 1e-6)
% b                     = abs(grad_b - grad_b1);
% find (b >1e-6)
training_loss           = zeros(n_epochs,1);
validation_loss         = zeros(n_epochs,1);

for i=1:n_epochs
    [W,b]                   = MiniBatchGD(trainX,trainY,n_batch,eta,n_epochs,W,b,lambda);
    t_cost                  = ComputeCost(trainX,trainY,W,b,lambda);
    training_loss(i,1)      = t_cost;
    v_cost                  = ComputeCost(valX,valY,W,b,lambda);
    validation_loss(i,1)    = v_cost;
%     if rem(i,n)==0
%         eta               = eta/10;
%     end
end
yline = 1:n_epochs;
figure()
a1 = plot(yline,training_loss,'g-');
M1 = 'Training loss ';
hold on
a2 = plot(yline,validation_loss,'b-');
M2 = 'Validation loss ';
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
ComputeAccuracy(testX,testy,W,b)




