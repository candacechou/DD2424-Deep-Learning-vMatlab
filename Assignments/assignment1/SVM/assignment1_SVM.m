s                        = rng(400);
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[valX, valY, valy]       = LoadBatch('data_batch_2.mat');

d                        = 32 * 32 * 3;
N                        = 10000;
K                        = 10;
rng(s);

W                        = 0.01 * randn(K,d);
b                        = 0.01 * randn(K,1);
lambda                   = 0.01;
n_batch                  = 400;
eta                      = 0.05;
n_epochs                 = 100;
C                        = 1;
training_loss            = zeros(n_epochs,1);
validation_loss          = zeros(n_epochs,1);
for i=1:n_epochs
    [W,b]                   = MiniBatchSVM(trainX,trainy,trainY,n_batch,eta,n_epochs,W,b,lambda);
    t_cost                  = ComputeCostSVM(trainX,trainY,W,b,lambda,C);
    training_loss(i,1)      = t_cost;
    v_cost                  = ComputeCostSVM(valX,valY,W,b,lambda,C);
    validation_loss(i,1)    = v_cost;
    
    eta = eta * 0.9;
    
end
yline           = 1:n_epochs;
figure()
a1              = plot(yline,training_loss,'g-');
M1              = 'Training loss ';
hold on
a2              = plot(yline,validation_loss,'b-');
M2              = 'Validation loss ';
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
