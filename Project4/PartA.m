PCA = importdata('PCAscores.mat')
LDA = importdata('LDAscores.mat')
mainMin = []
mainMax = []
mainAvg = []


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


for c = 1:200
    for r = 1:200
        mainMax(c,r)=max(PCA(c,r),LDA(c,r));
    end
end

for c = 1:200
    for r = 1:200
        mainMin(c,r)=min(PCA(c,r),LDA(c,r));
    end
end

for c = 1:200
    for r = 1:200 
        mainAvg(c,r)=((PCA(c,r)+LDA(c,r))/2);
    end
end



%Utilize ezroc function to evaluate performance
avgPlot =  ezroc3(mainAvg,labelsMain,2,'',1);
minPlot =  ezroc3(mainMin,labelsMain,2,'',1);
maxPlot =  ezroc3(mainMax,labelsMain,2,'',1);
LDAPlot =  ezroc3(LDA,labelsMain,2,'',1);
PCAPlot =  ezroc3(PCA,labelsMain,2,'',1);
mat1 = []
mat2 = []
for c = 1:503
    mat1(1,c)=avgPlot(1,c);
    mat2(2,c)=avgPlot(2,c);
end

mat3 = []
mat4 = []
for c = 1:503
    mat3(1,c)=minPlot(1,c);
    mat4(2,c)=minPlot(2,c);
end

mat5 = []
mat6 = []
for c = 1:503
    mat5(1,c)=maxPlot(1,c);
    mat6(2,c)=maxPlot(2,c);
end

mat7 = []
mat8 = []
for c = 1:503
    mat7(1,c)=LDAPlot(1,c);
    mat8(2,c)=LDAPlot(2,c);
end

mat9 = []
mat10 = []
for c = 1:503
    mat9(1,c)=PCAPlot(1,c);
    mat10(2,c)=PCAPlot(2,c);
end


plot(mat2,mat1,'color','r')
hold on
plot(mat4,mat3,'color','b')
hold on
plot(mat6,mat5,'color','g')
hold on
plot(mat8,mat7,'color','y')
hold on
plot(mat10,mat9,'color','c')
hold off

legend('\color{red} Average','\color{blue} Minimum','\color{green} Max', '\color{yellow} LDA', '\color{cyan} PCA')


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