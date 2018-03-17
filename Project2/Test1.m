vertArray2 = [];
compare1 = [];
compare2 = [];
for i = 26:40
    g1 = [];
    for j = 1:10
        %y = dir(strcat('att_faces\s',i,'.pgm'));
        r = strcat('att_faces\s',int2str(i));
        y1 = strcat(r,'\',int2str(j),'.pgm');
        v = imread(y1);
        r = reshape(v', [], 1);
        g1 = horzcat(g1,r);
    end
    compare1 = horzcat(compare1,g1);
end
for i = 26:40
    g2 = [];
    for j = 6:10
        %y = dir(strcat('att_faces\s',i,'.pgm'));
        r = strcat('att_faces\s',int2str(i));
        y = strcat(r,'\',int2str(j),'.pgm');
        v = imread(y);
        r = reshape(v', [], 1);
        g2= horzcat(g2,r);
    end
    compare2 = horzcat(compare2,g2);
end
q= pdist2(compare1,compare2,'Euclidean');