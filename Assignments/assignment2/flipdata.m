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
    clear A; clear I;
end