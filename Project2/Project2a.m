
l = t(3).name;
vertArray = [];
for i = 1:40
    g = [];
    for j = 1:10
        
        %y = dir(strcat('att_faces\s',i,'.pgm'));
        r = strcat('att_faces\s',int2str(i));
        y = strcat(r,'\',int2str(j),'.pgm');
        v = imread(y);
        r = reshape(v', [], 1);
        g = vertcat(g,r);
    end
    vertArray = horzcat(vertArray,g);
end
H = resize(vertArray',103040,25);
%A = imread('name1.pgm');
%AColumnVector = reshape(A', [], 1);
%GColumnVector = reshape(g', [], 1);

