%data extraction using imageset
Data = imageSet('att_faces','recursive');
train_data=cell(1,250);
test_data=cell(1,150);
a = 1;
for j=1:25
 for i=1:10 %all 10 images of all 25 subjects for training
 X= read(Data(j),i);
 X=reshape(X,prod(size(X)),1);
 X=double(X);
 train_data{a} = X;
 a = a + 1;
 end;
end;
a=1;
 for j=1:15 %15 subjects
 for i=1:10 %all 10 images of 15 subjects for testing
 X= read(Data(j),i);
 X=reshape(X,prod(size(X)),1);
 X=double(X);
 test_data{a} = X;
 a = a + 1;
 end;
end;
%%converting the cellarray to ordinary array or matrix
train_data=cell2mat(train_data);
test_data=cell2mat(test_data);
% Compute the mean of the data matrix
m=mean(train_data,2); %for the training set
% Subtract the mean from each image [Centering the data]
d=train_data-repmat(m,1,250); %for the training set
% Compute the covariance matrix (co)
co=d*d';
% Compute the eigen values and eigen vectors of the covariance matrix
[eigvector,eigvl]=eig(co);
% Sort the eigen vectors according to the eigen values
eigvalue = diag(eigvl);
[junk, index] = sort(eigvalue,'descend');
% Compute the number of eigen values that greater than zero (you can select
any threshold)
count1=0;
for i=1:size(eigvalue,1)
 if(eigvalue(i)>0)
 count1=count1+1;
 end
end
% And also we can use the eigen vectors that the corresponding eigen values
is greater than zero(Threshold) and this method will decrease the
% computation time and complixity
vec=eigvector(:,index(1:250));
%%projection
tr_pro=vec'*d; %train projection
test_data=test_data-repmat(mean(test_data,2),1,150);% performing the mean of
the test matrix and subtracting the mean from each image(centering the data)
ts_pro=vec'*test_data; %test projection
%Use Euclidean distance as distance metrics.
D=pdist2(tr_pro',ts_pro','Euclidean');
%labels
labels=zeros(250,150);
for i=1:250
 for j=1:150
 if(fix((i-1)/10)==fix((j-1)/10))
 labels(i,j)=0;
 else
 labels(i,j)=1;
 end
 end
end
ezroc3(D, labels,2,'',1);