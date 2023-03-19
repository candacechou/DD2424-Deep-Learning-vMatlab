
clc,clear all
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


%%%% Function for both with and without batch normalization
% 0. define parameter structures

%%%%%%%%%%%%%%%%%%%%%%%%%put this into other script %%%%%%%%%%%%%%%%%%%%%%
% classdef net_params
%     properties
%         use_bn          %%%% if you use Batch Normalization of not
%         W               %%%% weight
%         b               %%%% bias
%         Grad_W          %%%% gradients of weight
%         Grad_b          %%%% gradients of bias
%         gammas          %%%% gammas of all layers
%         betas           %%%% betas of all layers
%         Grad_gm         %%%% gradients of gammas
%         Grad_bt         %%%% gradients of betas
%         n_v             %%%% normalized variance  
%         n_mu            %%%% normalized mean
%         un_v            %%%% unnormalized variance
%         un_mu           %%%% unnormalized mean
%         ave_v           %%%% moving average variance
%         ave_mu          %%%% moving average mean
%         alpha           %%%% moving average parameters alpha
%     end
% end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. LoadBatch
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
% 2. Plot the results
function [ ] = Plot_results(t_loss,t_cost,t_acc,v_loss,v_cost,v_acc,yline,etas)
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
% ylim([1.5,3]);
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
hold off

figure()
a1 = plot(etas,'g-');
M1 = 'etas ';
legend(M1);
title(" etas");
xlabel(" update step");
ylabel("etas ");
hold off

figure()
a1 = plot(yline,t_cost,'g-');
M1 = 'Training loss ';
hold on
a2 = plot(yline,v_cost,'b-');
M2 = 'Validation loss ';
legend(M1,M2);
title(" Loss each update step");
xlabel(" update step  ");
ylabel(" Loss Function ");
% ylim([1.5,3]);
hold off
end
% 3. Compute the accuracy
function acc = ComputeAccuracy(X, y,net,k_layers,t,P)
    [~,ii] = max(P);
    ii = reshape(ii,size(ii,2),1);
    acc = size(find(ii==y),1) / size(ii,1);
    
end
% 4. Compute Cost and loss
function [J,loss] = ComputeCost(X, Y, net, lambda,k_layers,P,matrix) 
        %%%% compute cost and loss with matrix : 0, with for loop :1
        %%%% for faster computation time
            J         = 0;
            [d,D]     = size(X);
            if matrix == 0
            l         = -log(Y'*P); %% nxn
            
%             J         = (1/D)*sum(l,'all');
             J         = (1/D) * sum(diag(l),'all');
            
            else 
                for j=1:D/1000
                    j_start           = (j-1) * 1000 + 1;
                    j_end             = j * 1000;    
                    inds              = j_start:j_end; 
                    Pbatch            = P(:,inds);
                    Ybatch            = Y(:,inds);
                    l = -log(Ybatch' * Pbatch);
                    J = J + trace(l);
                end
                J = J/D;
            end
            loss      = J;
        for i=1:k_layers
            J       = J + lambda* sum(sum(net.W{i,1}.^2));
        end

end
% 5. update weight 
function net = updateweight(net,eta,k_layers,BN)
    for i=1:k_layers
        net.W{i}      = net.W{i} - eta * net.Grad_W{i};
        net.b{i}      = net.b{i} - eta * net.Grad_b{i};
        if BN == 1 && i<k_layers
            net.gammas{i}        = net.gammas{i} - eta * net.Grad_gm{i};
            net.betas{i}         = net.betas{i} - eta * net.Grad_bt{i};         
    end
    end
end
%%%% functions without batch normalization
% 1. Initialize paramters
function [net] = InitializeParameter(X,k_layers,hidden_layers)
    [d,n]           = size(X);
    net             = net_params;
    net.use_bn      = 0;
    net.W           = cell(k_layers,1);
    net.Grad_W      = cell(k_layers,1);
    net.Grad_b      = cell(k_layers,1);
    net.b           = cell(k_layers,1);
    Xinit           = sqrt(2)/sqrt(d);
    net.W{1,1}      = Xinit * randn(hidden_layers(1),d);
    net.Grad_W{1,1} = zeros(hidden_layers(1),d);
    net.b{1,1}      = zeros(hidden_layers(1),1);
    net.Grad_b{1,1} = zeros(hidden_layers(1),1);
    for i=1:k_layers-1
        if i ~= k_layers-1
            Xinit               = sqrt(2)/sqrt(hidden_layers(i));
            net.W{i+1,1}        = Xinit * randn(hidden_layers(i+1),hidden_layers(i));
            net.b{i+1,1}        = zeros(hidden_layers(i+1),1);
            net.Grad_W{i+1,1}   = zeros(hidden_layers(i+1),hidden_layers(i));
            net.Grad_b{i+1,1}   = zeros(hidden_layers(i+1),1);
        else
            Xinit               = sqrt(2)/sqrt(hidden_layers(i));
            net.W{i+1,1}        = Xinit * randn(10,hidden_layers(i));
            net.b{i+1,1}        = zeros(10,1);
            net.Grad_W{i+1,1}   = zeros(10,hidden_layers(i));
            net.Grad_b{i+1,1}   = zeros(10,1);
        end
    end
end

% 
% 2. compute gradients numerically
function [grad_b, grad_W] = ComputeGradsNumSlowOld(X, Y, net, lambda, h,k_layers)
W      = net.W;
b      = net.b;
grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        net1   = net;
        net1.b = b_try;
        c1 = ComputeCost(X, Y, net1,lambda,k_layers);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        net1.b = b_try;
        c2 = ComputeCost(X, Y, net1, lambda,k_layers);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        net2  = net;
      
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        net2.W = W_try;
        c1 = ComputeCost(X, Y, net2, lambda,k_layers);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        net2.W = W_try;
        c2 = ComputeCost(X, Y, net2, lambda,k_layers);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end

% 3. compute gradients analytically
function [net] = ComputeGradients(X, Y, net , lambda,k_layers,P,h)
    [K,n]       = size(Y);
    [d,n]       = size(X);
    %[P,h,s]     = EvaluateClassifier(X,net,k_layers);
    g           = -(Y-P);
    for i = k_layers:-1:2
        net.Grad_b{i,1}   = sum(g,2)./n;
        net.Grad_W{i,1}   = g * h{i-1,1}'./n + 2 * lambda * net.W{i,1};
        g                 = net.W{i,1}' * g;
        indX              = h{i-1,1};
        indX(indX < 0)    = 0;
        indX(indX > 0)    = 1;
        g                 = g .* indX;
    end
    %%% finally :
    
    net.Grad_W{1,1}       = g * X'./n;
    net.Grad_b{1,1}       = sum(g,2)./n;
        
        
        
end
% 4. checking gradients


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
end

% 5. Evaluate classifier
function [P,h,s_sum] = EvaluateClassifier(X,net,k_layers)
        [d,n]   = size(X);
        k       = 10;
        x_temp  = X;
        j       = k_layers - 1;
        h       = cell(j,1);
        s_sum   = cell(j,1);
        for i=1:k_layers-1
            b_m        = repmat(net.b{i,1},1,n);
            s          = net.W{i,1} * x_temp + b_m;
            x_temp     = max(0,s);
            h{i,1}     = x_temp;
            s_sum{i,1} = s;
        end
        b_m               = repmat(net.b{k_layers,1},1,n);
        s                 = net.W{k_layers,1} * x_temp + b_m;
        s_sum{k_layers,1} = s;
        P                 = exp(s);
        Psum              = sum(P,1);
        Psum              = repmat(Psum,k,1);
        P                 = P ./Psum; 
            
        
        
end

%%%% functions for batch normalization
% 1. Initialize paramters
function [net] = InitializeParameter_BN(X,k_layers,hidden_layers,alpha)
    [d,n]           = size(X);
    net             = net_params;
    net.use_bn      = 1;
    net.W           = cell(k_layers,1);
    net.Grad_W      = cell(k_layers,1);
    net.Grad_b      = cell(k_layers,1);
    net.b           = cell(k_layers,1);
    net.betas       = cell(k_layers-1,1);
    net.gammas      = cell(k_layers-1,1);
    net.Grad_gm     = cell(k_layers-1,1);
    net.Grad_bt     = cell(k_layers-1,1);
    net.n_v         = cell(k_layers,1);
    net.n_mu        = cell(k_layers,1);
    net.un_v        = cell(k_layers,1);
    net.un_mu       = cell(k_layers,1);
    net.ave_v       = cell(k_layers,1);
    net.ave_mu      = cell(k_layers,1);
    net.alpha       = alpha;
    %%%%%% initialize betas and gammas and their gradients
    Xinit           = sqrt(2)/sqrt(d);
    net.W{1,1}      = Xinit * randn(hidden_layers(1),d);
    net.Grad_W{1,1} = zeros(hidden_layers(1),d);
    net.b{1,1}      = zeros(hidden_layers(1),1);
    net.Grad_b{1,1} = zeros(hidden_layers(1),1);
   
    for i=1:k_layers-1
        if i ~= k_layers-1
            Xinit               = sqrt(2)/sqrt(hidden_layers(i));
            net.W{i+1,1}        = Xinit * randn(hidden_layers(i+1),hidden_layers(i));
            net.b{i+1,1}        = zeros(hidden_layers(i+1),1);
            net.Grad_W{i+1,1}   = zeros(hidden_layers(i+1),hidden_layers(i));
            net.Grad_b{i+1,1}   = zeros(hidden_layers(i+1),1);
        else
            Xinit               = sqrt(2)/sqrt(hidden_layers(i));
            net.W{i+1,1}        = Xinit * randn(10,hidden_layers(i));
            net.b{i+1,1}        = zeros(10,1);
            net.Grad_W{i+1,1}   = zeros(10,hidden_layers(i));
            net.Grad_b{i+1,1}   = zeros(10,1);
        end
    end
end

function [net] = Initial_gamma_betas(net,k_layers,hidden_layers,batch_size,alpha)
    net.alpha       = alpha;
    net.betas       = cell(k_layers-1,1);
    net.gammas      = cell(k_layers-1,1);
    for i=1:k_layers-1
        net.betas{i}    = zeros(hidden_layers(i),1);
        net.gammas{i}   = ones(hidden_layers(i),1);
    end
end

% 2. Compute gradients analytically
function [net] = ComputeGradients_BN(X, Y, net , lambda,k_layers,P,Xhat_batch,s,s_hat)
    [K,n]                   = size(Y);
    [d,n]                   = size(X);
    g                       = -(Y-P);
    net.Grad_b{k_layers}    = sum(g,2)./n;
    net.Grad_W{k_layers}    = g * Xhat_batch{k_layers}'./n + 2 * lambda * net.W{k_layers,1};
    g                       = net.W{k_layers}' * g;
    indX                    = Xhat_batch{k_layers};
    indX(indX < 0)          = 0;
    indX(indX > 0)          = 1;
    g                       = g .* indX;
    for i = k_layers-1:-1:1
        
            [col,n]                 = size(g);
%             I_n                     = ones(n);
            net.Grad_gm{i}          = sum(g .* s_hat{i},2)./n;
            net.Grad_bt{i}          = sum(g,2)./n;
            g                       = g.*repmat(net.gammas{i},1,n);
            %%%% batchNormBackPass
            g                       = BatchNormBackPass(g,s{i},net.un_mu{i},net.un_v{i});
            %%%% calculate the gradient of bias and weights
            
            net.Grad_b{i}           = sum(g,2)./n;
            net.Grad_W{i}           = g * Xhat_batch{i}'./n + 2 * lambda * net.W{i};
            %%%% calculate new G_batch
            if i > 1
                g                       = net.W{i}' * g;
                indX                    = Xhat_batch{i};
                indX(indX < 0)          = 0;
                indX(indX > 0)          = 1;
                g                       = g .* indX;
                
            end
        end
    %%% finally the first layer of gradient W and b :
    
%     net.Grad_W{1}           = g * X'./n;
%     net.Grad_b{1}           = sum(g,2)./n;
%         
        
        
end
% 3. Compute gradients numerically
function Grads = ComputeGradsNumSlow_option1(X, Y, NetParams, lambda, h,k_layers,t)

Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    Grads.gammas = cell(numel(NetParams.gammas), 1);
    Grads.betas = cell(numel(NetParams.betas), 1);
end

for j=1:length(NetParams.b)
    Grads.b{j,1} = zeros(size(NetParams.b{j,1}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
        c1 = ComputeCost(X, Y, NetTry, lambda,k_layers,t);        
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
        c2 = ComputeCost(X, Y, NetTry, lambda,k_layers,t);
        
        Grads.b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    Grads.W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
        c1 = ComputeCost(X, Y, NetTry, lambda,k_layers,t);
    
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
        c2 = ComputeCost(X, Y, NetTry, lambda,k_layers,t);
    
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gammas)
        Grads.gammas{j} = zeros(size(NetParams.gammas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gammas{j})
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) - h;
            NetTry.gammas = gammas_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda,k_layers,t);
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) + h;
            NetTry.gammas = gammas_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda,k_layers,t);
            
            Grads.gammas{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.betas)
        Grads.betas{j} = zeros(size(NetParams.betas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.betas{j})
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) - h;
            NetTry.betas = betas_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda,k_layers,t);
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) + h;
            NetTry.betas = betas_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda,k_layers,t);
            
            Grads.betas{j}(i) = (c2-c1) / (2*h);
        end
    end    
end
end
% 4. checking gradients

function [] = CheckingGradient_BN(net,grad_BN,k_layers,h)
    for i = 1:k_layers
        fprintf('the %d th layers\n',i);
        if i <= k_layers-1
            
            Wa = abs(net.Grad_W{i,1} - grad_BN.W{i,1});
            fprintf('W');
            find(Wa > h)
            Wb = abs(net.Grad_b{i,1} - grad_BN.b{i,1});
            fprintf('b');
            find(Wb > h)
            Wg = abs(net.Grad_gm{i,1} - grad_BN.gammas{i,1});
            fprintf('gamma');
            find(Wg > h)
            Wbeta = abs(net.Grad_bt{i,1} - grad_BN.betas{i,1});
            fprintf('betas');
            find(Wbeta > h)
        else
            fprintf('the %d th layers\n',i);
            Wa = abs(net.Grad_W{i,1} - grad_BN.W{i,1});
            fprintf('W');
            find(Wa > h)
            Wb = abs(net.Grad_b{i,1} - grad_BN.b{i,1});
            fprintf('b');
            find(Wb > h)
        end
    end
end

% 5. Evaluate classifier
function [P,Xhat_batch,s,s_hat,net] = EvaluateClassifier_BN(X,net,k_layers,t,cost)
        [d,n]            = size(X);
        k                = 10;
        x_temp           = X;
        j                = k_layers - 1;
        Xhat_batch       = cell(k_layers,1);
        Xhat_batch{1,1}  = x_temp;
        s                = cell(j,1);
        s_hat            = cell(j,1);
        for i=1:k_layers-1
            %b_m             = repmat(net.b{i,1},1,n);
            s{i,1}              = net.W{i,1} * x_temp + net.b{i,1};
            %%%% not sure assuming make average on each dimension
            net.un_mu{i,1}      = sum(s{i,1},2)/size(s{i,1},2);
            %net.un_mu{i,1}  = mean(s{i,1},2);
            net.un_v{i,1}       = sum((s{i,1} -net.un_mu{i,1}).^2,2)./size(s{i,1},2);
%             net.ave_mu{i,1}     = net.un_mu{i,1};
%             net.ave_v{i,1}      = net.un_v{i,1};
            
            %%%% Batch Normalization
            if cost == 0
                s_hat{i,1}          = Batch_Normalization(s{i,1},net.un_mu{i,1},net.un_v{i,1});
                if t == 1
                    net.ave_mu{i,1} = net.un_mu{i,1};
                    net.ave_v{i,1}  = net.un_v{i,1};
                else
            %%%% do the moving average
                    net.ave_mu{i,1}     = net.alpha * net.ave_mu{i,1} + (1 - net.alpha) * net.un_mu{i,1};
                    net.ave_v{i,1}      = net.alpha * net.ave_v{i,1} + (1 - net.alpha) * net.un_v{i,1};
                end
            else
                s_hat{i,1}          = Batch_Normalization(s{i,1},net.ave_mu{i,1},net.ave_v{i,1});
            end
            s_shave             = repmat(net.gammas{i},1,n) .*  s_hat{i,1} + net.betas{i};
            x_temp              = max(0,s_shave);
            Xhat_batch{i+1,1}   = x_temp;
        end
        s_temp            = net.W{k_layers,1} * x_temp + net.b{k_layers,1};
        P                 = exp(s_temp);
        Psum              = sum(P,1);
        Psum              = repmat(Psum,k,1);
        P                 = P ./Psum;       
end

