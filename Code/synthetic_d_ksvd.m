function [acc] = synthetic_d_ksvd(meandifference)
% specify dictionary dimensions

n = 504;
atoms = 2; % m

% specify parameters
subject = 2;
images_persubject = 32;
support = 16;

iteration = 50;
alpha = 1;
%meandifference=0.1;

%%

Y=zeros(n,subject*images_persubject);
for i=0:subject-1
    for j=1:images_persubject
        Y(:,i*images_persubject+j)=normc(randn(n,1))+i*meandifference;

    end
end
%%
T=zeros(n,subject*images_persubject);
for i=0:subject-1
    for j=1:images_persubject
        T(:,i*images_persubject+j)=normc(randn(n,1))+i*meandifference;

    end
end
TestL=zeros(subject*images_persubject,1);
for i=0:subject-1
    TestL(i*images_persubject+1:i*images_persubject+images_persubject)=i+1;
end

%%
%set up parameters for ksvd training
ksvds_params.data = Y;
ksvds_params.Tdata = support;
ksvds_params.dictsize = atoms;
ksvds_params.iternum = 20;
ksvds_params.memusage = 'high';


% k-svd toolbox => initialize dictionary
[D,gamma,err] = ksvds(ksvds_params,'');

% initialize classifier;
I = diag(ones(size(D, 2)));
I = diag(I);

% H = the label of the training images
H = zeros(subject, subject*images_persubject);
for i = 1 : subject
    H(i, (i-1) * images_persubject + 1 : i * images_persubject) = 1;
end

% the weight for classify and represent
Data = [alpha*H; Y];  % the first matrix in Eqn. 10

% W = the parameter of the classifier
W = ((gamma*gamma'+I)\gamma*H')';
dictionary = [alpha*W; D];  % the second matrix in Eqn. 10

% start training
disp('start training');
for i=1:iteration
    disp(['Iter: ', num2str(i)]);
    
    % find the coefficient_matrix (Dictionary, coef) = argmin||[Y; alpha*H]-[D; alpha*W]*coef||2
    coefficient_matrix = omp(dictionary, Data, support);
    rdm = randperm(size(dictionary,2));

    % update Dictionary and coefficient_matrix
    for j = rdm
        [better_element,coefficient_matrix] = Better_Dictionary_Element(Data, dictionary, j, coefficient_matrix);
        dictionary(:,j) = better_element;
    end
end
%%
% normalize the dictionary and classifier . % Eqn. 14
W = dictionary(1: size(W,1), :);
new_dict = dictionary(size(W,1)+1: size(dictionary,1), :);
for i=1 : size(new_dict, 2)
    dict_norm = norm(new_dict(:, i));
    new_dict(:, i) = new_dict(:, i)/dict_norm;
    W(:, i) = W(:, i)/dict_norm;
end
W = W/alpha;

disp('training complete');
% SAVE dict, W, coefficient_matrix
save('d_ksvd_training_result.mat', 'new_dict', 'W', 'coefficient_matrix');
%%
test_params.data=T;
test_id=zeros(1, size(test_params.data,2));

t1=cputime;
for i=1:size(test_params.data,2)
    alpha=omp(new_dict, T(:,i), 64);
    l=W*alpha;
    [c,test_id(i)]=max(abs(l));
end
t2=cputime-t1;
%%
count=0;
for i=1:length(test_id)
    if test_id(i)==TestL(i)
        count=count+1;
    end
end
acc=count/length(test_id);
end