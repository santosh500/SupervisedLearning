G = [
1	2	3	1	0
1	2	1	0	0
1	4	3	1	2
1	3	4	2	0
1	0	0	0	0
];

ncols = size(G,2);
OUT = cell(ncols,1);
for nc = 1:ncols
    OUT{nc} = fliplr(G(:,1:nc));
    OUT{nc}(any(OUT{nc}==0,2),:) = [];
end

a=OUT{3,1}

QMAN = cell(ncols,1);

trainPhotos = [];
testPhotos = [];
for i = 1:40
    g = [];
    for j = 1:5
        %y = dir(strcat('att_faces\s',i,'.pgm'));
        r = strcat('att_faces\s',int2str(i));
        y = strcat(r,'\',int2str(j),'.pgm');
        v = imread(y);
        r = v(:);
        g = horzcat(g,r);
    end
    QMAN{i} = double(g);
    trainPhotos = horzcat(trainPhotos,g);
end
trainData = double(trainPhotos);

featureCount = 10304;

sample1 = QMAN{1,1};
sample2 = QMAN{2,1};
sample3 = QMAN{3,1};
sample4 = QMAN{4,1};
sample5 = QMAN{5,1};
sample6 = QMAN{6,1};
sample7 = QMAN{7,1};
sample8 = QMAN{8,1};
sample9 = QMAN{9,1};
sample10 = QMAN{10,1};
sample11 = QMAN{11,1};
sample12 = QMAN{12,1};
sample13 = QMAN{13,1};
sample14 = QMAN{14,1};
sample15 = QMAN{15,1};
sample16 = QMAN{16,1};
sample17 = QMAN{17,1};
sample18 = QMAN{18,1};
sample19 = QMAN{19,1};
sample20 = QMAN{20,1};
sample21 = QMAN{21,1};
sample22 = QMAN{22,1};
sample23 = QMAN{23,1};
sample24 = QMAN{24,1};
sample25 = QMAN{25,1};
sample26 = QMAN{26,1};
sample27 = QMAN{27,1};
sample28 = QMAN{28,1};
sample29 = QMAN{29,1};
sample30 = QMAN{30,1};
sample31 = QMAN{31,1};
sample32 = QMAN{32,1};
sample33 = QMAN{33,1};
sample34 = QMAN{34,1};
sample35 = QMAN{35,1};
sample36 = QMAN{36,1};
sample37 = QMAN{37,1};
sample38 = QMAN{38,1};
sample39 = QMAN{39,1};
sample40 = QMAN{40,1};


mu1=mean(sample1);
mu2=mean(sample2);
mu3=mean(sample3);
mu4=mean(sample4);
mu5=mean(sample5);
mu6=mean(sample6);
mu7=mean(sample7);
mu8=mean(sample8);
mu9=mean(sample9);
mu10=mean(sample10);
mu11=mean(sample11);
mu12=mean(sample12);
mu13=mean(sample13);
mu14=mean(sample14);
mu15=mean(sample15);
mu16=mean(sample16);
mu17=mean(sample17);
mu18=mean(sample18);
mu19=mean(sample19);
mu20=mean(sample20);
mu21=mean(sample21);
mu22=mean(sample22);
mu23=mean(sample23);
mu24=mean(sample24);
mu25=mean(sample25);
mu26=mean(sample26);
mu27=mean(sample27);
mu28=mean(sample28);
mu29=mean(sample29);
mu30=mean(sample30);
mu31=mean(sample31);
mu32=mean(sample32);
mu33=mean(sample33);
mu34=mean(sample34);
mu35=mean(sample35);
mu36=mean(sample36);
mu37=mean(sample37);
mu38=mean(sample38);
mu39=mean(sample39);
mu40=mean(sample40);


overallTotal = (mu1+mu2+mu3+mu4+mu5+mu6+mu7+mu8+mu9+mu10+mu11+mu12+mu13+mu14+mu15+mu16+mu17+mu18+mu19+mu20+mu21+mu22+mu23+mu24+mu25+mu26+mu27+mu28+mu29+mu30+mu31+mu32+mu33+mu34+mu35+mu36+mu37+mu38+mu39+mu40);
totalSamples = 40;
mu = overallTotal/totalSamples;

d1 = sample1-repmat(mu1,size(sample1,1),1);
d2 = sample2-repmat(mu2,size(sample2,1),1);
d3 = sample3-repmat(mu3,size(sample3,1),1);
d4 = sample4-repmat(mu4,size(sample4,1),1);
d5 = sample5-repmat(mu5,size(sample5,1),1);
d6 = sample6-repmat(mu6,size(sample6,1),1);
d7 = sample7-repmat(mu7,size(sample7,1),1);
d8 = sample8-repmat(mu8,size(sample8,1),1);
d9 = sample9-repmat(mu9,size(sample9,1),1);
d10 = sample10-repmat(mu10,size(sample10,1),1);
d11 = sample11-repmat(mu11,size(sample11,1),1);
d12 = sample12-repmat(mu12,size(sample12,1),1);
d13 = sample13-repmat(mu13,size(sample13,1),1);
d14 = sample14-repmat(mu14,size(sample14,1),1);
d15 = sample15-repmat(mu15,size(sample15,1),1);
d16 = sample16-repmat(mu16,size(sample16,1),1);
d17 = sample17-repmat(mu17,size(sample17,1),1);
d18 = sample18-repmat(mu18,size(sample18,1),1);
d19 = sample19-repmat(mu19,size(sample19,1),1);
d20 = sample20-repmat(mu20,size(sample20,1),1);
d21 = sample21-repmat(mu21,size(sample21,1),1);
d22 = sample22-repmat(mu22,size(sample22,1),1);
d23 = sample23-repmat(mu23,size(sample23,1),1);
d24 = sample24-repmat(mu24,size(sample24,1),1);
d25 = sample25-repmat(mu25,size(sample25,1),1);
d26 = sample26-repmat(mu26,size(sample26,1),1);
d27 = sample27-repmat(mu27,size(sample27,1),1);
d28 = sample28-repmat(mu28,size(sample28,1),1);
d29 = sample29-repmat(mu29,size(sample29,1),1);
d30 = sample30-repmat(mu30,size(sample30,1),1);
d31 = sample31-repmat(mu31,size(sample31,1),1);
d32 = sample32-repmat(mu32,size(sample32,1),1);
d33 = sample33-repmat(mu33,size(sample33,1),1);
d34 = sample34-repmat(mu34,size(sample34,1),1);
d35 = sample35-repmat(mu35,size(sample35,1),1);
d36 = sample36-repmat(mu36,size(sample36,1),1);
d37 = sample37-repmat(mu37,size(sample37,1),1);
d38 = sample38-repmat(mu38,size(sample38,1),1);
d39 = sample39-repmat(mu39,size(sample39,1),1);
d40 = sample40-repmat(mu40,size(sample40,1),1);

s1=d1'*d1;
s2=d2'*d2;
s3=d3'*d3;
s4=d4'*d4;
s5=d5'*d5;
s6=d6'*d6;
s7=d7'*d7;
s8=d8'*d8;
s9=d9'*d9;
s10=d10'*d10;
s11=d11'*d11;
s12=d12'*d12;
s13=d13'*d13;
s14=d14'*d14;
s15=d15'*d15;
s16=d16'*d16;
s17=d17'*d17;
s18=d18'*d18;
s19=d19'*d19;
s20=d20'*d20;
s21=d21'*d21;
s22=d22'*d22;
s23=d23'*d23;
s24=d24'*d24;
s25=d25'*d25;
s26=d26'*d26;
s27=d27'*d27;
s28=d28'*d28;
s29=d29'*d29;
s30=d30'*d30;
s31=d31'*d31;
s32=d32'*d32;
s33=d33'*d33;
s34=d34'*d34;
s35=d35'*d35;
s36=d36'*d36;
s37=d37'*d37;
s38=d38'*d38;
s39=d39'*d39;
s40=d40'*d40;


sw = (s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15+s16+s17+s18+s19+s20+s21+s22+s23+s24+s25+s26+s27+s28+s29+s30+s31+s32+s33+s34+s35+s36+s37+s38+s39+s40);
invsw=inv(sw)

sb1=10304*(mu1-mu)'*(mu1-mu)
sb2=10304*(mu2-mu)'*(mu2-mu)
sb3=10304*(mu3-mu)'*(mu3-mu)
sb4=10304*(mu4-mu)'*(mu4-mu)
sb5=10304*(mu5-mu)'*(mu5-mu)
sb6=10304*(mu6-mu)'*(mu6-mu)
sb7=10304*(mu7-mu)'*(mu7-mu)
sb8=10304*(mu8-mu)'*(mu8-mu)
sb9=10304*(mu9-mu)'*(mu9-mu)
sb10=10304*(mu10-mu)'*(mu10-mu)
sb11=10304*(mu11-mu)'*(mu11-mu)
sb12=10304*(mu12-mu)'*(mu12-mu)
sb13=10304*(mu13-mu)'*(mu13-mu)
sb14=10304*(mu14-mu)'*(mu14-mu)
sb15=10304*(mu15-mu)'*(mu15-mu)
sb16=10304*(mu16-mu)'*(mu16-mu)
sb17=10304*(mu17-mu)'*(mu17-mu)
sb18=10304*(mu18-mu)'*(mu18-mu)
sb19=10304*(mu19-mu)'*(mu19-mu)
sb20=10304*(mu20-mu)'*(mu20-mu)
sb21=10304*(mu21-mu)'*(mu21-mu)
sb22=10304*(mu22-mu)'*(mu22-mu)
sb23=10304*(mu23-mu)'*(mu23-mu)
sb24=10304*(mu24-mu)'*(mu24-mu)
sb25=10304*(mu25-mu)'*(mu25-mu)
sb26=10304*(mu26-mu)'*(mu26-mu)
sb27=10304*(mu27-mu)'*(mu27-mu)
sb28=10304*(mu28-mu)'*(mu28-mu)
sb29=10304*(mu29-mu)'*(mu29-mu)
sb30=10304*(mu30-mu)'*(mu30-mu)
sb31=10304*(mu31-mu)'*(mu31-mu)
sb32=10304*(mu32-mu)'*(mu32-mu)
sb33=10304*(mu33-mu)'*(mu33-mu)
sb34=10304*(mu34-mu)'*(mu34-mu)
sb35=10304*(mu35-mu)'*(mu35-mu)
sb36=10304*(mu36-mu)'*(mu36-mu)
sb37=10304*(mu37-mu)'*(mu37-mu)
sb38=10304*(mu38-mu)'*(mu38-mu)
sb39=10304*(mu39-mu)'*(mu39-mu)
sb40=10304*(mu40-mu)'*(mu40-mu)

sb = (sb1+sb2+sb3+sb4+sb5+sb6+sb7+sb8+sb9+sb10+sb11+sb12+sb13+sb14+sb15+sb16+sb17+sb18+sb19+sb20+sb21+sb22+sb23+sb24+sb25+sb26+sb27+sb28+sb29+sb30+sb31+sb32+sb33+sb34+sb35+sb36+sb37+sb38+sb39+sb40);
v=invsw*(sb);

[evec,eval]=eig(v)

y1=sample1*v;
y2=sample2*v;
y3=sample3*v;
y4=sample4*v;
y5=sample5*v;
y6=sample6*v;
y7=sample7*v;
y8=sample8*v;
y9=sample9*v;
y10=sample10*v;
y11=sample11*v;
y12=sample12*v;
y13=sample13*v;
y14=sample14*v;
y15=sample15*v;
y16=sample16*v;
y17=sample17*v;
y18=sample18*v;
y19=sample19*v;
y20=sample20*v;
y21=sample21*v;
y22=sample22*v;
y23=sample23*v;
y24=sample24*v;
y25=sample25*v;
y26=sample26*v;
y27=sample27*v;
y28=sample28*v;
y29=sample29*v;
y30=sample30*v;
y31=sample31*v;
y32=sample32*v;
y33=sample33*v;
y34=sample34*v;
y35=sample35*v;
y36=sample36*v;
y37=sample37*v;
y38=sample38*v;
y39=sample39*v;
y40=sample40*v;

y2=cat(2,y1,y2);
y3=cat(2,y2,y3);
y4=cat(2,y3,y4);
y5=cat(2,y4,y5);
y6=cat(2,y5,y6);
y7=cat(2,y6,y7);
y8=cat(2,y7,y8);
y9=cat(2,y8,y9);
y10=cat(2,y9,y10);
y11=cat(2,y10,y11);
y12=cat(2,y11,y12);
y13=cat(2,y12,y13);
y14=cat(2,y13,y14);
y15=cat(2,y14,y15);
y16=cat(2,y15,y16);
y17=cat(2,y16,y17);
y18=cat(2,y17,y18);
y19=cat(2,y18,y19);
y20=cat(2,y19,y20);
y21=cat(2,y20,y21);
y22=cat(2,y21,y22);
y23=cat(2,y22,y23);
y24=cat(2,y23,y24);
y25=cat(2,y24,y25);
y26=cat(2,y25,y26);
y27=cat(2,y26,y27);
y28=cat(2,y27,y28);
y29=cat(2,y28,y29);
y30=cat(2,y29,y30);
y31=cat(2,y30,y31);
y32=cat(2,y31,y32);
y33=cat(2,y32,y33);
y34=cat(2,y33,y34);
y35=cat(2,y34,y35);
y36=cat(2,y35,y36);
y37=cat(2,y36,y37);
y38=cat(2,y37,y38);
y39=cat(2,y38,y39);
y40=cat(2,y39,y40);



TMAN = cell(ncols,1);
for i = 1:40
    g = [];
    for j = 6:10
        %y = dir(strcat('att_faces\s',i,'.pgm'));
        r = strcat('att_faces\s',int2str(i));
        y = strcat(r,'\',int2str(j),'.pgm');
        val = imread(y);
        r = val(:);
        g = horzcat(g,r);
    end
    TMAN{i} = double(g);
    testPhotos = horzcat(testPhotos,g);
end
testData = double(testPhotos);

test1 = TMAN{1,1};
test2 = TMAN{2,1};
test3 = TMAN{3,1};
test4 = TMAN{4,1};
test5 = TMAN{5,1};
test6 = TMAN{6,1};
test7 = TMAN{7,1};
test8 = TMAN{8,1};
test9 = TMAN{9,1};
test10 = TMAN{10,1};
test11 = TMAN{11,1};
test12 = TMAN{12,1};
test13 = TMAN{13,1};
test14 = TMAN{14,1};
test15 = TMAN{15,1};
test16 = TMAN{16,1};
test17 = TMAN{17,1};
test18 = TMAN{18,1};
test19 = TMAN{19,1};
test20 = TMAN{20,1};
test21 = TMAN{21,1};
test22 = TMAN{22,1};
test23 = TMAN{23,1};
test24 = TMAN{24,1};
test25 = TMAN{25,1};
test26 = TMAN{26,1};
test27 = TMAN{27,1};
test28 = TMAN{28,1};
test29 = TMAN{29,1};
test30 = TMAN{30,1};
test31 = TMAN{31,1};
test32 = TMAN{32,1};
test33 = TMAN{33,1};
test34 = TMAN{34,1};
test35 = TMAN{35,1};
test36 = TMAN{36,1};
test37 = TMAN{37,1};
test38 = TMAN{38,1};
test39 = TMAN{39,1};
test40 = TMAN{40,1};

tu1=mean(test1);
tu2=mean(test2);
tu3=mean(test3);
tu4=mean(test4);
tu5=mean(test5);
tu6=mean(test6);
tu7=mean(test7);
tu8=mean(test8);
tu9=mean(test9);
tu10=mean(test10);
tu11=mean(test11);
tu12=mean(test12);
tu13=mean(test13);
tu14=mean(test14);
tu15=mean(test15);
tu16=mean(test16);
tu17=mean(test17);
tu18=mean(test18);
tu19=mean(test19);
tu20=mean(test20);
tu21=mean(test21);
tu22=mean(test22);
tu23=mean(test23);
tu24=mean(test24);
tu25=mean(test25);
tu26=mean(test26);
tu27=mean(test27);
tu28=mean(test28);
tu29=mean(test29);
tu30=mean(test30);
tu31=mean(test31);
tu32=mean(test32);
tu33=mean(test33);
tu34=mean(test34);
tu35=mean(test35);
tu36=mean(test36);
tu37=mean(test37);
tu38=mean(test38);
tu39=mean(test39);
tu40=mean(test40);

testTotal = (tu1+tu2+tu3+tu4+tu5+tu6+tu7+tu8+tu9+tu10+tu11+tu12+tu13+tu14+tu15+tu16+tu17+tu18+tu19+tu20+tu21+tu22+tu23+tu24+tu25+tu26+tu27+tu28+tu29+tu30+tu31+tu32+tu33+tu34+tu35+tu36+tu37+tu38+tu39+tu40);
totalSamples = 40;
tu = testTotal/totalSamples;

x1=test1*v;
x2=test2*v;
x3=test3*v;
x4=test4*v;
x5=test5*v;
x6=test6*v;
x7=test7*v;
x8=test8*v;
x9=test9*v;
x10=test10*v;
x11=test11*v;
x12=test12*v;
x13=test13*v;
x14=test14*v;
x15=test15*v;
x16=test16*v;
x17=test17*v;
x18=test18*v;
x19=test19*v;
x20=test20*v;
x21=test21*v;
x22=test22*v;
x23=test23*v;
x24=test24*v;
x25=test25*v;
x26=test26*v;
x27=test27*v;
x28=test28*v;
x29=test29*v;
x30=test30*v;
x31=test31*v;
x32=test32*v;
x33=test33*v;
x34=test34*v;
x35=test35*v;
x36=test36*v;
x37=test37*v;
x38=test38*v;
x39=test39*v;
x40=test40*v;
 
x2=cat(2,x1,x2);
x3=cat(2,x2,x3);
x4=cat(2,x3,x4);
x5=cat(2,x4,x5);
x6=cat(2,x5,x6);
x7=cat(2,x6,x7);
x8=cat(2,x7,x8);
x9=cat(2,x8,x9);
x10=cat(2,x9,x10);
x11=cat(2,x10,x11);
x12=cat(2,x11,x12);
x13=cat(2,x12,x13);
x14=cat(2,x13,x14);
x15=cat(2,x14,x15);
x16=cat(2,x15,x16);
x17=cat(2,x16,x17);
x18=cat(2,x17,x18);
x19=cat(2,x18,x19);
x20=cat(2,x19,x20);
x21=cat(2,x20,x21);
x22=cat(2,x21,x22);
x23=cat(2,x22,x23);
x24=cat(2,x23,x24);
x25=cat(2,x24,x25);
x26=cat(2,x25,x26);
x27=cat(2,x26,x27);
x28=cat(2,x27,x28);
x29=cat(2,x28,x29);
x30=cat(2,x29,x30);
x31=cat(2,x30,x31);
x32=cat(2,x31,x32);
x33=cat(2,x32,x33);
x34=cat(2,x33,x34);
x35=cat(2,x34,x35);
x36=cat(2,x35,x36);
x37=cat(2,x36,x37);
x38=cat(2,x37,x38);
x39=cat(2,x38,x39);
x40=cat(2,x39,x40);


D=pdist2(y40',x40','Euclidean');





labels=zeros(200,200);
for i=1:200
 for j=1:200
 if(fix((i-1)/5)==fix((j-1)/5))
 labels(i,j)=0;
 else
 labels(i,j)=1;
 end
 end
end
ezroc3(D,labels,2,'',1);


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