diretory2data='C:\Users\jianfen6\Desktop\project\project\facedata';
currentFolder = pwd;
% specify dictionary dimensions
n = 504;
atoms = 304; % m

% specify parameters
subject = 38;
images_persubject = 32;
support = 16;

iteration = 50;
alpha = 1;
%%
Randomface = randn(n, 192*168);
for i = 1 : n
    Randomface(i, :) = Randomface(i, :) / norm(Randomface(i, :));
end
%%
%go to the directory and get a list of all images
cd(diretory2data);
eval('list=dir(''*.pgm'');');
disp('reading images...');
%read in the images and resize them . Y = input signal
image = imread(list(1).name);
image_size = size(image, 1) * size(image, 2);
tmp=str2num(list(1).name(6:7));


no_l=1;
num_each_sub=0;
for i= 1 : length(list)
    if(tmp~=str2num(list(i).name(6:7)))
        no_l=no_l+1;
        num_each_sub=1;
        param(no_l).label(num_each_sub)=no_l;
        image = imread(list(i).name);
        image=normc(im2double(image));
        param(no_l).data(:,num_each_sub)=Randomface*double(reshape(image, image_size, 1));
        tmp=str2num(list(i).name(6:7));
    else
        num_each_sub=num_each_sub+1;
        param(no_l).label(num_each_sub)=no_l;
        image = imread(list(i).name);
        image=normc(im2double(image));
        param(no_l).data(:,num_each_sub)=Randomface*double(reshape(image, image_size, 1));
        
    end
end
%%
disp('set training set and testing set');
trainnum=1;
testnum=1;
for i=1:subject
    v=randperm(length(param(i).label));
    for j=1:32
        Y(:,trainnum)=param(i).data(:,v(j));
        TrainL(:,trainnum)=param(i).label(v(j));
        trainnum=trainnum+1;
    end
    for j=33:length(v)
        T(:,testnum)=param(i).data(:,v(j));
        TestL(:,testnum)=param(i).label(v(j));
        testnum=testnum+1;
    end
end
cd (currentFolder);

%%
disp('Initializing with K-SVD');
%set up parameters for ksvd training
ksvds_params.data = Y;
ksvds_params.Tdata = support;
ksvds_params.dictsize = atoms;
ksvds_params.iternum = 20;
ksvds_params.memusage = 'high';

init1=cputime;
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
initialtime=cputime-init1;
%%
% start training
disp('start training');
tr1=cputime;
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
traintime=cputime-tr1;
%%
disp('nomalizing dictionary and classifier')
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
disp('start testing');
test_params.data=T;
test_id=zeros(1, size(test_params.data,2));

t1=cputime;
for i=1:size(test_params.data,2)
    alpha=omp(new_dict, T(:,i), 64);
    l=W*alpha;
    [c,test_id(i)]=max(abs(l));
end
testtime=cputime-t1;
%%
count=0;
for i=1:length(test_id)
    if test_id(i)==TestL(i)
        count=count+1;
    end
end
acc=count/length(test_id);
X = sprintf('The accuracy is %f',acc);
disp(X)
X = sprintf('Time cost for initialing: %f',initialtime);
disp(X)
X = sprintf('Time cost for training: %f',traintime);
disp(X)
X = sprintf('Time cost for testing: %f',testtime);
disp(X)
