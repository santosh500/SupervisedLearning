%import PCA and LDA matrices
PCA = importdata('PCAscores.mat')
LDA = importdata('LDAscores.mat')
GeneralLabels = importdata('LabelsGeneral.mat')
%initialize matrix for combined matrices
PCACombine = [];
LDACombine = [];

%Create the PCA Multi-Instance matrix
zeroMatrix = zeros(5);
oneMatrix = ones(5);
total = 0;
round1 = 1;
round2 = 1;
for i=1:200 
    increment = 0;
    for j=1:200
        total = total + PCA(j,i);
        increment = increment + 1;
        if(increment==5)
            PCACombine(round1,round2) = (total/5);
            increment = 0;
            round1 = round1+1;
            total = 0;
        end
    end
    round1=1;
    round2 = round2+1;
end


%Create the LDA Multi-Instance matrix
total = 0;
round1 = 1;
round2 = 1;
for i=1:200 
    increment = 0;
    for j=1:200
        total = total + LDA(j,i);
        increment = increment + 1;
        if(increment==5)
            LDACombine(round1,round2) = (total/5);
            increment = 0;
            round1 = round1+1;
            total = 0;
        end
    end
    round1=1;
    round2 = round2+1;
end


%Create Labels
labelsOfficial = [];
labelsMain = []
labelCount = 1;
regularZero = [0 0 0 0 0];
regularOne = [1 1 1 1 1];
for i=1:40 
    labelsOfficial = [];
    for j=1:40
        if(labelCount==j)
            labelsOfficial = horzcat(labelsOfficial,regularZero);
        else
            labelsOfficial = horzcat(labelsOfficial,regularOne);
        end
    end
    labelsMain = vertcat(labelsMain,labelsOfficial);
    labelCount = labelCount + 1;
end

%Utilize ezroc function to evaluate performance for PCA_MutiInstance,LDA_MultiInstance,LDA,PCA
PCAMIPlot =  ezroc3(PCACombine,labelsMain,2,'',1);
LDAMIPlot =  ezroc3(LDACombine,labelsMain,2,'',1);
LDAPlot =  ezroc3(LDA,GeneralLabels,2,'',1);
PCAPlot =  ezroc3(PCA,GeneralLabels,2,'',1);

%Obtain x and y for min
PCAMIX = []
PCAMIY = []
for c = 1:503
    PCAMIX(1,c)=PCAMIPlot(1,c);
    PCAMIY(2,c)=PCAMIPlot(2,c);
end

%Obtain x and y for max
LDAMIX = []
LDAMIY = []
for c = 1:503
    LDAMIX(1,c)=LDAMIPlot(1,c);
    LDAMIY(2,c)=LDAMIPlot(2,c);
end

%Obtain x and y for LDA plot
ldaX = []
ldaY = []
for c = 1:503
    ldaX(1,c)=LDAPlot(1,c);
    ldaY(2,c)=LDAPlot(2,c);
end

%Obtain x and y for PCA plot
PCAX = []
PCAY = []
for c = 1:503
    PCAX(1,c)=PCAPlot(1,c);
    PCAY(2,c)=PCAPlot(2,c);
end

%Create fusion plot
plot(PCAMIY,PCAMIX,'color','r')
title('Min')
hold on
plot(LDAMIY,LDAMIX,'color','b')
title('Max')
hold on
plot(ldaY,ldaX,'color','g')
title('LDA')
hold on
plot(PCAY,PCAX,'color','y')
hold off

legend('\color{red} PCA Fusion','\color{blue} LDA Fusion','\color{green} LDA Single', '\color{yellow} PCA Single')
title('Multi-Instance vs Single Classifier')
xlabel('FAR')
ylabel('GAR')

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