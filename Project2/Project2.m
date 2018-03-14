A = imread('name1.pgm')
t = dir('att_faces');
fields = fieldnames(t);
ls=[];
for i = 1:numel(fields)
  t.(fields{2});
  
end
b = dir('att_faces');
b(1)
names =  fieldnames(b(1));
p = size(t);
for i = 5:p
  u = strcat(t(i).folder,t(i).name);
 %ls(end+1) = u;
end
%access a struct 

l = t(3).name;
g = [];
for i = 1:10
    %y = dir(strcat('att_faces\s',i,'.pgm'));
    r = strcat('att_faces\s',int2str(i));
    y = strcat(r,'\',int2str(i),'.pgm');
    v = imread(y);
    g = horzcat(g,v);
end

data = double(imread('name1.pgm'));   
data = double(g);
[r,c] = size(data);
%data = int16(1:r);   
% Compute the mean of the data matrix "The mean of each row"
m = mean(data')';
% Subtract the mean from each image [Centering the data]
k2 = repmat(m,1,c);
d=data-repmat(m,1,c);


% Compute the covariance matrix (co)
co=d*d';

% Compute the eigen values and eigen vectors of the covariance matrix
[eigvector,eigvl]=eig(co);


% Sort the eigen vectors according to the eigen values
eigvalue = diag(eigvl);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

% Compute the number of eigen values that greater than zero (you can select any threshold)
count1=0;
for i=1:size(eigvalue,1)
    if(eigvalue(i)>0)
        count1=count1+1;
    end
end

% We can use all the eigen vectors but this method will increase the
% computation time and complixity
%vec=eigvector(:,:);

% And also we can use the eigen vectors that the corresponding eigen values is greater than zero(Threshold) and this method will decrease the
% computation time and complixity

vec=eigvector(:,1:count1);

% Compute the feature matrix (the space that will use it to project the testing image on it)
x=vec'*d;

