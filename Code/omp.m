function [coefficient_matrix]=omp(dictionary,signals,atoms) 

for k = 1 : size(signals, 2)
    r = signals(:,k);
    index = zeros(atoms,1);
    for j = 1 : atoms
        proj = dictionary'*r;
        [value,position] = max(abs(proj));
        index(j) = position(1);
        a = pinv(dictionary(:,index(1:j)))*signals(:,k);
        r = signals(:,k) - dictionary(:,index(1:j))*a;
        if sum(r.^2) < 1e-6
            break;
        end
    end
    temp = zeros(size(dictionary, 2),1);
    temp(index(1:j)) = a;
    coefficient_matrix(:,k) = sparse(temp);
end
