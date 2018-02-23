A = importdata('featureMat_liv_train_bioLBP.mat');
B = importdata('featureMat_Latex_train_bioLBP.mat');
C = importdata('featureMat_Gelatine_train_bioLBP.mat');
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

%Step 4
realData = importdata('featureMat_liv_train_bioLBP.mat');
gelatineData = importdata('featureMat_Gelatine_train_bioLBP.mat');
realData(:,1) = [];
testLive=predict(Mdl,realData);
testGelatine=predict(Mdl,realData);



totalMatrix = vertcat(A,B,C);
%Q=outerjoin(A,B);

disp(V);
str = strings(2,3);


totalGoal = {};
for i=1:1000
    totalGoal{end+1} = 'real';
end
for i=1:400
    totalGoal{end+1} = 'fake';
end
trainLabels = reshape(totalGoal,1400,1);

disp(totalGoal);
kuBBall= totalMatrix(:,2:54);
tabulate(trainLabels)


Md2 = fitcnb(kuBBall,trainLabels,...
    'ClassNames',{'real','fake'});

realDataTrain = importdata('featureMat_liv_train_bioLBP.mat');
realDataTrain(:,1) = [];
gelatineDataTrain = importdata('featureMat_Gelatine_train_bioLBP.mat');
gelatineDataTrain(:,1) = [];
latexDataTrain = importdata('featureMat_Latex_train_bioLBP.mat');

%P1 = importdata('featureMat_liv_train_bioLBP.mat');
%P1(:,1) = [];

predictReal=predict(Md2,realDataTrain);
predictGelatine=predict(Md2,gelatineDataTrain);
fakeList = {};
for i=1:200
    fakeList{end+1} = 'fake';
end
fakeReshape = reshape(fakeList,200,1);

realList= {};
for i=1:1000
    realList{end+1} = 'real';
end
realReshape = reshape(realList,1000,1);

%#5
lossReal = loss(Md2,realDataTrain,realReshape);
lossGelatine= loss(Md2,gelatineDataTrain,fakeReshape);
rsLoss4 = resubLoss(Md2);

%#6

prior = [0.6 0.4];
Md3 = fitcnb(X2,y,'ClassNames',{'real','fake'},'Prior',prior)

labelPriorLive=predict(Md3,P);
labelPriorLatex=predict(Md3,P1);
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
lossFakePrior = loss(Md3,P,K2);
lossRealPrior = loss(Md3,P1,K3);
rsLossPrior = resubLoss(Md3);

