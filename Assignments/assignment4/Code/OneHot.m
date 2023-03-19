% function Y = OneHot(book_data,book_char)
% 
% input :
%     book_data  = 1xn %%% how many words we have
%     book_char  = 1xk %%% how many characters form this book
%     
% output : 
%     Y          = kxn %%% one hot representation
%     
%     
function Y = OneHot(book_data,book_char)
    [~,n] = size(book_data);
    [~,k] = size(book_char);
    Y = zeros(k,n);
    for i=1:k
        [~,ii] = find(book_data == book_char(i));
        Y(i,ii) = 1;
    end
end
