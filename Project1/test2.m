A = importdata('featureMat_liv_train_bioLBP.mat');
B = importdata('featureMat_Latex_train_bioLBP.mat');
disp('Hello World.');
%disp(A);
disp(B);
%Mdl = fitcnb(A,'Texas');

%X2= B(:,2:54);
%disp(X2)
V = vertcat(A,B);
%Q=outerjoin(A,B);

%disp(V);
%str = strings(2,3);


names = {};
for i=1:1000
    names{end+1} = 'real';
end
for i=1:200
    names{end+1} = 'fake';
end
y = reshape(names,1200,1);

disp(names);
X2 = V(:,2:54);


Mdl = fitcnb(X2,y,...
    'ClassNames',{'real','fake'});

setosaIndex = strcmp(Mdl.ClassNames,'real');
estimates = Mdl.DistributionParameters{setosaIndex,1}
P = importdata('featureMat_Latex_train_bioLBP.mat');
P(:,1) = [];

P = importdata('featureMat_Latex_train_bioLBP.mat');
P(:,1) = [];

P1 = importdata('featureMat_liv_train_bioLBP.mat');
P1(:,1) = [];

label=predict(Mdl,P);
label2=predict(Mdl,P1);
u = 0;
for i=1:199
    if isequal(label(i,1),'real')
        u=u+1;
    end
    disp(label(i,1))
end
disp(u)
