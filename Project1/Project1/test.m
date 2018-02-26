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

disp(V);
str = strings(2,3);


names = {};
for i=1:1000
    names{end+1} = 'real';
end
for i=1:200
    names{end+1} = 'fake';
end
y = reshape(names,1200,1);

disp(names);
X2= V(:,2:54);
tabulate(y)


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
D = {};
for i=1:200
    D{end+1} = 'fake';
end
K2 = reshape(D,200,1);

D2 = {};
for i=1:1000
    D2{end+1} = 'real';
end
K3 = reshape(D2,1000,1);
L = loss(Mdl,P,K2);
L2 = loss(Mdl,P1,K3);
rsLoss1 = resubLoss(Mdl);
