dis1=synthetic_d_ksvd(0.1);
dis2=synthetic_d_ksvd(0.2);
dis3=synthetic_d_ksvd(0.8);
dis4=synthetic_d_ksvd(1);
dis5=synthetic_d_ksvd(10);
dis6=synthetic_d_ksvd(100);
%%
X = sprintf('When the difference of mean is 0.1, accuracy is %f',dis1);
disp(X)
X = sprintf('When the difference of mean is 0.2, accuracy is %f',dis2);
disp(X)
X = sprintf('When the difference of mean is 0.8, accuracy is %f',dis3);
disp(X)
X = sprintf('When the difference of mean is 1, accuracy is %f',dis4);
disp(X)
X = sprintf('When the difference of mean is 10, accuracy is %f',dis5);
disp(X)
X = sprintf('When the difference of mean is 100, accuracy is %f',dis6);
disp(X)
