clc,clear all
rng shuffle;
[trainX, trainY, trainy]    = LoadBatch('data_batch_1.mat');
[ValX, ValY, Valy]          = LoadBatch('data_batch_2.mat');
[testX, testY, testy]       = LoadBatch('test_batch.mat');
%[Flip_X,Flip_Y,Flip_y] = flipdata('data_batch_1.mat');
[trainX1, trainY1, trainy1] = LoadBatch('data_batch_1.mat');
[trainX2, trainY2, trainy2] = LoadBatch('data_batch_2.mat');
[trainX3, trainY3, trainy3] = LoadBatch('data_batch_3.mat');
[trainX4, trainY4, trainy4] = LoadBatch('data_batch_4.mat');
[trainX5, trainY5, trainy5] = LoadBatch('data_batch_5.mat');
[testX, testY, testy]       = LoadBatch('test_batch.mat');
trainX                      = [trainX1,trainX2,trainX3,trainX4,trainX5(:,1:9000)];
trainY                      = [trainY1,trainY2,trainY3,trainY4,trainY5(:,1:9000)];
trainy                      = [trainy1;trainy2;trainy3;trainy4;trainy5(1:9000)];
ValX                        = trainX5(:,9001:end);
ValY                        = trainY5(:,9001:end);
Valy                        = trainy5(9001:end);
[d,n]                       = size(trainX);
f                           = 20;
l_min                       = -3;
l_max                       = -1;
k_num                       = 10;
m                           = 200;
std_1                       = 1/sqrt(d);
std_2                       = 1/sqrt(m);
mean_1                      = 0 ;
mean_2                      = 0;
[W,b, lambda]               = InitializeParameter(trainX,k_num,std_1,mean_1,std_2,mean_2,l_min,l_max,m);
%%%% testing the gradient 
[h,P]                       = EvaluateClassifier(trainX, W,b);
[grad_W,grad_b]             = ComputeGradients(trainX, trainY, P,h,W,b,lambda);
[grad_b_test, grad_W_test]  = ComputeGradsNumSlow(trainX, trainY, W, b, lambda, 1e-5);
%%% testing the gradient function
%%% - testing -> absolute only
W1_a                        = abs(grad_W{1} - grad_W_test{1});
find(W1_a > 1e-6)
W2_a                        = abs(grad_W{2} - grad_W_test{2});
find(W2_a > 1e-6)
b1_a                        = abs(grad_b{1} - grad_b_test{1});
find(b1_a > 1e-6)
b2_a                        = abs(grad_b{2} - grad_b_test{2});
find(b2_a > 1e-6)
% % %%%% - testing -> with more accurate 
% % h                       = 1e-6;
% % W1_a                    = abs(grad_W{1} - grad_W_test{1})./max(h,abs(grad_W{1})+abs(grad_W_test{1}));
% % find(W1_a > 1e-4)
% % W2_a                    = abs(grad_W{2} - grad_W_test{2})./max(h,abs(grad_W{2})+abs(grad_W_test{2}));
% % find(W2_a > 1e-4)
% % b1_a                    = abs(grad_b{1} - grad_b_test{1})./max(h,abs(grad_b{1})+abs(grad_b_test{1}));
% % find(b1_a > 1e-4)
% % b2_a                    = abs(grad_b{2} - grad_b_test{2})./max(h,abs(grad_b{2})+abs(grad_b_test{2}));
% % find(b2_a > 1e-4)
