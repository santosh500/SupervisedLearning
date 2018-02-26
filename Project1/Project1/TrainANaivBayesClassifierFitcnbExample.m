%% Train a Naive Bayes Classifier
%%
% Load Fisher's iris data set.
load fisheriris

X = meas(:,3:4);
disp(X)
Y = species;
tabulate(Y)
%%
% The software can classify data with more than two classes
% using naive Bayes methods.
%%
% Train a naive Bayes classifier.  It is good practice to specify the class
% order.
Mdl = fitcnb(X,Y,...
    'ClassNames',{'setosa','versicolor','virginica'})
%%
% |Mdl| is a trained |ClassificationNaiveBayes| classifier.
%%
% By default, the software models the predictor distribution within each class
% using a Gaussian distribution having some mean and standard deviation. Use
% dot notation to display the parameters of a particular Gaussian fit,
% e.g., display the fit for the first feature within |setosa|.
setosaIndex = strcmp(Mdl.ClassNames,'setosa');
estimates = Mdl.DistributionParameters{setosaIndex,1}
%%
% The mean is |1.4620| and the standard deviation is |0.1737|.
%%
% Plot the Gaussian contours.
figure
gscatter(X(:,1),X(:,2),Y);
h = gca;
cxlim = h.XLim;
cylim = h.YLim;
hold on
Params = cell2mat(Mdl.DistributionParameters); 
Mu = Params(2*(1:3)-1,1:2); % Extract the means
Sigma = zeros(2,2,3);
for j = 1:3
    Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
    xlim = Mu(j,1) + 4*[1 -1]*sqrt(Sigma(1,1,j));
    ylim = Mu(j,2) + 4*[1 -1]*sqrt(Sigma(2,2,j));
    ezcontour(@(x1,x2)mvnpdf([x1,x2],Mu(j,:),Sigma(:,:,j)),[xlim ylim])
        % Draw contours for the multivariate normal distributions 
end
h.XLim = cxlim;
h.YLim = cylim;
title('Naive Bayes Classifier -- Fisher''s Iris Data')
xlabel('Petal Length (cm)')
ylabel('Petal Width (cm)')
hold off
%%
% You can change the default distribution using the name-value pair
% argument |'DistributionNames'|.  For example, if some predictors are
% categorical, then you can specify that they are multivariate, multinomial
% random variables using |'DistributionNames','mvmn'|.