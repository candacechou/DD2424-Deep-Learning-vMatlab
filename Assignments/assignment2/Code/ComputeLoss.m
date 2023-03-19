% function loss = ComputeLoss(X,Y,y,W,b)
%     input:
%         X = dxn
%         Y = Kxn
%         y = nx1
%         W =cell(1,2)
%         b = cell(1,2)
%     output:
%         loss = 1x1

function loss = ComputeLoss(X,Y,y,W,b)
loss = 0;
[h,P] = EvaluateClassifier(X,W,b);
loss_matrix = -log(Y'* P);
[d,D] = size(X);
loss = sum(diag(loss_matrix),'all')/D;

clear loss_matrix;


end