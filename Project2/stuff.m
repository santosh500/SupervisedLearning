for i=1:75
 for j=1:75
 if(fix((i-1)/5)==fix((j-1)/5))
 labels(i,j)=0;
 else
 labels(i,j)=1;
 end
 end
end

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