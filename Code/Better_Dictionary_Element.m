function [better_element,coeffitients] = Better_Dictionary_Element(Y,Dictionary,j,coeffitients)

if(isempty(find(coeffitients(j,:), 1)))
    [d,i] = max(sum((Y - Dictionary*coeffitients).^2));
    better_element = Y(:,i)./sqrt(Y(:,i)'*Y(:,i));
    better_element = better_element.*sign(better_element(1));
    coeffitients(j,:) = 0;
else
    index = find(coeffitients(j,:));
    temp_gamma = coeffitients(:,index); 
    temp_gamma(j,:) = 0;
    Ek =(Y(:,index) - Dictionary*temp_gamma); 
    [U,S,V] = svds(Ek,1);  % Eqn. 12
    better_element = U;
    coeffitients(j,index) = S*V';
end





