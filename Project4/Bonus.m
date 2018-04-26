clc
clear all
close all

%Get Data from ATT photo datatbase from subjects 1-40 for images 1-5 for PCA feature space
%creation
trainPhotos = [];
for i = 1:40
    g = [];
    for j = 1:5
        r = strcat('att_faces\s',int2str(i));
        y = strcat(r,'\',int2str(j),'.pgm');
        v = imread(y);
        r = v(:);
        g = horzcat(g,r);
    end
    trainPhotos = horzcat(trainPhotos,g);
end
%PCA train data manipulation 
trainData = double(trainPhotos);
[r,c] = size(trainData);
m = mean(trainData')';
%Normalize PCA train data
d=trainData-repmat(m,1,c);
%Create covariance matrix
co=d*d';
%Obtain eigenvectors and eigenvalues from co-variance matrices
[eigvector,eigvl]=eig(co);
eigvalue = diag(eigvl);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);
%Count number of valid eigenvalues that are positive
count1=0;
for i=1:size(eigvalue,1)
    if(eigvalue(i)>0)
        count1=count1+1;
    end
end
%Obtain vector space
vec=eigvector(:,1:count1);

%Get testing data for performance valuation from ATT database from subjects 1-40 for images 6-10 for PCA feature space
%creation
testPhotos = [];
for i = 1:40
    g = [];
    for j = 6:10
        %y = dir(strcat('att_faces\s',i,'.pgm'));
        r = strcat('att_faces\s',int2str(i));
        y = strcat(r,'\',int2str(j),'.pgm');
        v = imread(y);
        r = v(:);
        g = horzcat(g,r);
    end
    testPhotos = horzcat(testPhotos,g);
end
%PCA testing data manipulation 
testData = double(testPhotos);
testData=(testData-m);

%Multiply vector space by train and test data for performance evaluation
train=vec'*d;
test = vec'*testData;
singleFeatures = 5
cellArray = cell(singleFeatures,1);
classes = 40;
featureCount = 10304;
g=[]
totalNumber = 1;
testPhotos = [];
for i = 1:200
    trainPhotos = [];
    for j = 1:4974
        %y = dir(strcat('att_faces\s',i,'.pgm'));
        z = train(j,i);
        trainPhotos = vertcat(trainPhotos,z);
    end
    g = horzcat(g,trainPhotos);
    if(mod(i,5)==0)
        cellArray{totalNumber} = double(g);
        totalNumber = totalNumber + 1;
        g=[];
    end
end


%Obtain the average of the training data in a 1x5 matrix
totalMean = 0;
for i=1:classes
    tempMean = double(mean(cellArray{i,1}));
    totalMean = totalMean +tempMean;
end
average = (totalMean/classes);

%Obtain the SW or scatter-within from the train data
SW = 0;
for i=1:classes
    sample = cellArray{i,1};
    sampleMean = mean(sample);
    d = sample-repmat(sampleMean,size(sample,1),1);
    sw=d'*d;
    SW = SW + sw;
end
%Inverse the the SW
invsw=inv(SW);

%Obtain the SB or scatter-between from the train data
SB = 0;
for i=1:classes
    sample = cellArray{i,1};
    sampleMean = mean(sample);
    sb = featureCount*(sampleMean-average)'*(sampleMean-average);
    SB = SB + sb;
end
%Obtain vector space by utilizing SB/SW
v=invsw*(SB);

%Obtain the eigenvalues and eigenvectors from the eigenspace
[evec,eval]=eig(v);

%Project the training data into the vector space
trainVec = [];
for i=1:classes
    sample = cellArray{i,1};
    sampleVec = sample*v;
    trainVec = cat(2,trainVec,sampleVec);
end



testNumber = 1;
%Obtain the test data for images 6-10 for all classes
testArray = cell(singleFeatures,1);
for i = 1:200
    testPhotos = [];
    for j = 1:4974
        %y = dir(strcat('att_faces\s',i,'.pgm'));
        z = test(j,i);
        testPhotos = vertcat(testPhotos,z);
    end
    g = horzcat(g,testPhotos);
    if(mod(i,5)==0)
        testArray{testNumber} = double(g);
        testNumber = testNumber + 1;
        g=[];
    end
end
testData = double(testPhotos);

%Project the test data into the vector space
testVec = [];
for i=1:classes
    test = testArray{i,1};
    sampleVec = test*v;
    testVec = cat(2,testVec,sampleVec);
end

%Obtain the distance difference from an Eucliden minimum distance
distance=pdist2(testVec',trainVec','Euclidean');


%Create the label matrix
zeroMatrix = zeros(5);
oneMatrix = ones(5);
labelsMain = []
labelCount = 1;
generalMat = [];
for i=1:40 
    generalMat = [];
    for j=1:40
        if(labelCount==j)
            generalMat = horzcat(generalMat,zeroMatrix);
        else
            generalMat = horzcat(generalMat,oneMatrix);
        end
    end
    labelsMain = vertcat(labelsMain,generalMat);
    labelCount = labelCount + 1;
end

%Utilize the EZRoc function to obtain results
ezroc3(distance,labelsMain,2,'',1);


function [roc,EER,area,EERthr,ALLthr,d,gen,imp]=ezroc3(H,T,plot_stat,headding,printInfo)%,rbst
t1=min(min(min(H)));
t2=max(max(max(H)));
num_subj=size(H,1);

stp=(t2-t1)/500;   %step size here is 0.2% of threshold span, can be adjusted

if stp==0   %if all inputs are the same...
    stp=0.01;   %Token value
end
ALLthr=(t1-stp):stp:(t2+stp);
if (nargin==1 || (nargin==3 &&  isempty(T))||(nargin==2 &&  isempty(T))||(nargin==4 &&  isempty(T))||(nargin==5 &&  isempty(T)))  %Using only H, multi-class case, and maybe 3D or no plot
    GAR=zeros(503,size(H,3));  %initialize for accumulation in case of multiple H (on 3rd dim of H)
    FAR=zeros(503,size(H,3));
    gen=[]; %genuine scores place holder (diagonal of H), for claculation of d'
    imp=[]; %impostor scores place holder (non-diagonal elements of H), for claculation of d'
    for setnum=1:size(H,3); %multiple H measurements (across 3rd dim, where 2D H's stack up)
        gen=[gen; diag(H(:,:,setnum))]; %digonal scores
        imp=[imp; H(find(not(eye(size(H,2)))))]; %off-diagonal scores, with off-diagonal indices being listed by find(not(eye(size(H,2)))) 
        for t=(t1-stp):stp:(t2+stp),    %Note that same threshold is used for all H's, and we increase the limits by a smidgeon to get a full curve
            ind=round((t-t1)/stp+2);   %current loop index, +2 to start from 1
            id=H(:,:,setnum)>t;
            
            True_Accept=trace(id);  %TP
            False_Reject=num_subj-True_Accept;  %FN
            % In the following, id-diag(diag(id)) simply zeros out the diagonal of id
            True_Reject=sum( sum( (id-diag(diag(id)))==0 ) )-size(id,1); %TN, number of off-diag zeros. We need to subtract out the number of diagonals, as 'id-diag(diag(id))' introduces those many extra zeros into the sum
            False_Accept=sum( sum( id-diag(diag(id)) ) ); %FP, number of off-diagonal ones
            
            GAR(ind,setnum)=GAR(ind,setnum)+True_Accept/(True_Accept+False_Reject); %1-FRR, Denum: all the positives (correctly IDed+incorrectly IDed)
            FAR(ind,setnum)=FAR(ind,setnum)+False_Accept/(True_Reject+False_Accept); %1-GRR, Denum: all the negatives (correctly IDed+incorrectly IDed)
        end
    end
    GAR=sum(GAR,2)/size(H,3);   %average across multiple H's
    FAR=sum(FAR,2)/size(H,3);
elseif (nargin==2 || nargin==3 || nargin == 4 || nargin == 5),   %Regular, 1-class-vs-rest ROC, and maybe 3D or no plot
    gen=H(find(T)); %genuine scores
    imp=H(find(not(T))); %impostor scores
    for t=(t1-stp):stp:(t2+stp),    %span the limits by a smidgeon to get a full curve
        ind=round((t-t1)/stp+2);   %current loop index, +2 to start from 1
        id=H>t;
        
        True_Accept=sum(and(id,T)); %TP
        False_Reject=sum(and(not(id),T));   %FN
        
        True_Reject=sum(and(not(id),not(T)));   %TN
        False_Accept=sum(and(id,not(T)));   %FP
        
        GAR2(ind)=True_Accept/(True_Accept+False_Reject); %1-FRR, Denum: all the positives (correctly IDed+incorrectly IDed)
        FAR2(ind)=False_Accept/(True_Reject+False_Accept); %1-GRR, Denum: all the negatives (correctly IDed+incorrectly IDed)
        
    end
    GAR=GAR2';
    FAR=FAR2';
end
roc=[GAR';FAR'];
FRR=1-GAR;
[e ind]=min(abs(FRR'-FAR'));    %This is Approx w/ error e. Fix by linear inerpolation of neigborhood and intersecting w/ y=x
EER=(FRR(ind)+FAR(ind))/2;
area=abs(trapz(roc(2,:),roc(1,:)));
EERthr=t1+(ind-1)*stp;%EER threshold

d=abs(mean(gen)-mean(imp))/(sqrt(0.5*(var(gen)+var(imp))));   %Decidability or d'

if (nargin==1 || nargin==2 || nargin==3 || nargin == 4 || nargin == 5)
    if plot_stat == 2
        if printInfo == 1
            figure, plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),title(['ROC Curve: ' headding '   EER=' num2str(EER) ',   Area=' num2str(area) ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR');
        elseif printInfo == 0
            figure, plot(roc(2,:),roc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]),title(['ROC Curve: ' headding ' ']),xlabel('FAR'),ylabel('GAR');
        end
    elseif plot_stat == 3
        if printInfo == 1
            figure, plot3(roc(2,:),roc(1,:),ALLthr,'LineWidth',3),axis([0 1 0 1 (t1-stp) (t2+stp)]),title(['3D ROC Curve: ' headding '   EER=' num2str(EER) ',   Area=' num2str(area)  ',   Decidability=' num2str(d)]),xlabel('FAR'),ylabel('GAR'),zlabel('Threshold'),grid on,axis square;
        elseif printInfo == 0
            figure, plot3(roc(2,:),roc(1,:),ALLthr,'LineWidth',3),axis([0 1 0 1 (t1-stp) (t2+stp)]),title(['3D ROC Curve: ' headding ' ']),xlabel('FAR'),ylabel('GAR'),zlabel('Threshold'),grid on,axis square;
        end     
    else
        %else it must be 0, i.e. no plot
    end
end
end
