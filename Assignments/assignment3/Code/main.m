% clc,clear all
% rng shuffle;
% [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
% [ValX, ValY, Valy]       = LoadBatch('data_batch_2.mat');
% [testX, testY, testy]    = LoadBatch('test_batch.mat');

%%%%%% for larger dataset

[trainX1, trainY1, trainy1] = LoadBatch('data_batch_1.mat');
[trainX2, trainY2, trainy2] = LoadBatch('data_batch_2.mat');
[trainX3, trainY3, trainy3] = LoadBatch('data_batch_3.mat');
[trainX4, trainY4, trainy4] = LoadBatch('data_batch_4.mat');
[trainX5, trainY5, trainy5] = LoadBatch('data_batch_5.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');
trainX = [trainX1,trainX2,trainX3,trainX4,trainX5(:,1:5000)];
trainY = [trainY1,trainY2,trainY3,trainY4,trainY5(:,1:5000)];
trainy = [trainy1;trainy2;trainy3;trainy4;trainy5(1:5000)];
ValX = trainX5(:,5001:end);
ValY = trainY5(:,5001:end);
Valy = trainy5(5001:end);

%%%%%%%% parameters 

alpha                   = 0.9;
[d,n]                   = size(trainX);
% l_min                   = -5;
% l_max                   = -4;
k_num                   = 10;
% l = l_min + (l_max - l_min)*rand(1, 1); 
lambda =  0.0051;    
eta_min                 = 1e-3;
eta_max                 = 1e-1;
n_batch                 = 100;
ns                      = 5 * 45000 / n_batch;
k_layers                = 3;
hidden_layers           = [100,50];
epoch                   = 20;
kk                      = 45000/n_batch;
BN                      = 0; %%%% with batch normalization or not, 0: without; 1: with

%%%% checking the gradients
% net                     = InitializeParameter(trainX(1:200,1:200),k_layers,hidden_layers);
%%%% checking the gradients with numerical analysis

% [P,h,s_sum]             = EvaluateClassifier(trainX(1:200,1:200),net,k_layers);
% [grad_b, grad_W]        = ComputeGradsNumSlowOld(trainX(1:200,1:200), trainY(:,1:200), net, lambda, 1e-6,k_layers);
% net                     = ComputeGradients(trainX(1:200,1:200), trainY(:,1:200), net , lambda,k_layers,P,h);
% CheckingGradient(net,grad_W,grad_b,k_layers,1e-4)
% net                     = InitializeParameter_BN(trainX(1:200,1:200),k_layers,hidden_layers,n_batch);
% net                     = Initial_gamma_betas(net,k_layers,hidden_layers,200,alpha);

% [P, Xhat,s,s_hat,net]   = EvaluateClassifier_BN(trainX(1:200,1:200),net,k_layers,1);
% grad                    = ComputeGradsNumSlow_option1(trainX(1:200,1:200),trainY(:,1:200),net, lambda, 1e-6,k_layers,1);
% net                     = ComputeGradients_BN(trainX(1:200,1:200), trainY(:,1:200), net , lambda, k_layers,P,Xhat,s,s_hat);
% CheckingGradient_BN(net,grad,k_layers,1e-6)

%%%%

%%%% start to train
%%%% First of all, Initialize everything
if BN == 0
     net                     = InitializeParameter(trainX,k_layers,hidden_layers);
else
net                     = InitializeParameter_BN(trainX,k_layers,hidden_layers,n_batch);
net                     = Initial_gamma_betas(net,k_layers,hidden_layers,n_batch,alpha);
end 

t_loss                    = zeros(epoch,1);
v_loss                    = zeros(epoch,1);
t_cost                    = zeros(epoch,1);
v_cost                    = zeros(epoch,1);
t_acc                     = zeros(epoch,1);
v_acc                     = zeros(epoch,1);
etas                      = zeros(1000,1);
yline                     = zeros(epoch,1);
t                         = 1;
l                         = 0;

%%%% start to train

    for i = 1:epoch
            rand_i = randperm(n);
            shuffle_x     = trainX(:,rand_i);
            shuffle_y     = trainY(:,rand_i);
     for j=1:n/n_batch
        j_start           = (j-1) * n_batch + 1;
        j_end             = j * n_batch;    
        inds              = j_start:j_end; 
        Xbatch            = trainX(:,inds);
        Ybatch            = trainY(:,inds);
        if BN == 0 
            [P,h,s_sum]       = EvaluateClassifier(Xbatch, net,k_layers);
            net               = ComputeGradients(Xbatch, Ybatch, net , lambda,k_layers,P,h);
        else
            [P,Xhat_batch,s,s_hat,net] = EvaluateClassifier_BN(Xbatch,net,k_layers,t,0);
            net                        = ComputeGradients_BN(Xbatch, Ybatch, net , lambda,k_layers,P,Xhat_batch,s,s_hat);
        end
        
        %%%% etas calculation
        
        if (t >= 2 * l * ns) && (t < (2*l + 1) * ns)
            etas_unit     = eta_min + ((t - 2 * l * ns) / ns) * (eta_max - eta_min);
        elseif (t>=(2*l + 1) * ns) && t <(2*(l+1)*ns)
            etas_unit     = eta_max - ((t - (2 * l+1) * ns) / ns) * (eta_max - eta_min);
        end
    etas(t,1) = etas_unit;
    %%%% update the weights
    net               = updateweight(net,etas_unit,k_layers,BN);
   
    if rem(t,2*ns) == 0
         l            = l+1;
    end
    t
    t                 = t + 1;
    
     end
    
        if BN == 0
            [P,~,~]         = EvaluateClassifier(trainX, net,k_layers);
            [Pv,~,~]        = EvaluateClassifier(ValX, net,k_layers);
        else
            [P,~,~,~,~]     = EvaluateClassifier_BN(trainX,net,k_layers,t,1); 
            [Pv,~,~,~,~]    = EvaluateClassifier_BN(ValX,net,k_layers,t,1); 
        end
        [t_cost(i,1),t_loss(i,1)]     = ComputeCost(trainX, trainY, net, lambda,k_layers,P,1);
        t_acc(i,1)                    = ComputeAccuracy(trainX,trainy,net,k_layers,t,P);
        [v_cost(i,1),v_loss(i,1)]     = ComputeCost(ValX,ValY,net,lambda,k_layers,Pv,1);
        v_acc(i,1)                    = ComputeAccuracy(ValX,Valy,net,k_layers,t,Pv);
        yline(i,1)                    = t;
        i = i+1;

   
 end   
   

%%%% compute accuracy
if BN == 0
            [P,~,~]         = EvaluateClassifier(testX, net,k_layers);
            
        else
          [P,~,~,~,~]       = EvaluateClassifier_BN(testX,net,k_layers,t,1); 
          
        end
acc            = ComputeAccuracy(testX, testy,net,k_layers,t,P)

%%%% print 
 Plot_results(t_loss,t_cost,t_acc,v_loss,v_cost,v_acc,yline,etas);


