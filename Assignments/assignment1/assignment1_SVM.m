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
n_epochs              = 40;
C                     = 1;
training_loss         = zeros(n_epochs,1);
validation_loss       = zeros(n_epochs,1);
for i=1:n_epochs
    [W,b]                   = MiniBatchSVM(trainX,trainy,trainY,n_batch,eta,n_epochs,W,b,lambda);
    t_cost                  = ComputeCostSVM(trainX,trainY,W,b,lambda,C);
    training_loss(i,1)      = t_cost;
    v_cost                  = ComputeCostSVM(valX,valY,W,b,lambda,C);
    validation_loss(i,1)    = v_cost;
    if rem(i,10) == 0
    eta             = eta * 0.9;
    end
end
yline               = 1:n_epochs;
figure()
a1                  = plot(yline,training_loss,'g-');
M1                  = 'Training loss ';
hold on
a2                  = plot(yline,validation_loss,'b-');
M2                  = 'Validation loss ';
legend(M1,M2);
title(" Cost each Epoch");
xlabel(" Epoch ");
ylabel(" Cost Function ");
hold off

figure()
for i=1:10
    im              = reshape(W(i, :), 32, 32, 3);
    s_im{i}         = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i}         = permute(s_im{i}, [2, 1, 3]);
    subplot(2,5,i);
    imshow(s_im{i});
    
end
ComputeAccuracySVM(testX,testy,W,b,testY)
