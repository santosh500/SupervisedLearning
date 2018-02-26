
%% Compute the value of a Gaussian pdf at x1 = [0.2, 1.3] and x2=[2.2,-1.3]
%% where m= [0, 1]  and S= [1 0 ; 0 1]


m = [0 1]'; S = eye(2); %% setting mean and covariance
x1=[0.2 1.3]'; x2=[2.2 -1.3]';

pg1=mvnpdf(x1,m,S); %% Likelihood of x1 belonging to the distribution
pg2=mvnpdf(x2,m,S); %%% Likelihood of x2 belonging to the distribution





%%% Consider a 2-class classification task in the 2-dimensional space,
%%% where the data in both classes w1, w2, are distributed according to the
%%% Gaussian distributions N(m1,S1) and N(m2,S2), respectively, let
%%% m1=[1,1], m2=[3, 3], S1 = S2 = [1 0;0 1]. Assuming that P(w1)= P(w2)=
%%% 1/2, classify x = [1.8, 1.8] into w1 or w2.


P1 = 0.5; P2 = 0.5; %% setting Priors
m1=[1 1]'; m2=[3 3]'; %% initializing means of the two classes
S=eye(2); 
x = [1.8 1.8]'; %% test sample x initialized
scale= P1 * mvnpdf(x,m1,S)+ P2 * mvnpdf(x,m2,S);
post_w1 = (P1 * mvnpdf(x,m1,S))/scale;
post_w2 = (P2 * mvnpdf(x,m2,S))/scale;

%%% decision rule
if (post_w1> post_w2)
    decision = 1;
else
    decision = 2;
end
    





