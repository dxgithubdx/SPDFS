clc;
close all;

addpath('./'); addpath('./data'); addpath('./fcm'); addpath('./funs'); addpath('./SPDFS');

%% load data
load Coil20.mat
   
%% Preprocessing
fea = X; gnd = Y; c = max(Y);
num = size(fea,1);
fea = normalizefea(num, fea); 

%% Parameter setting
k = 50;      % number of selected features  50:50:300  
phi = 1.2;   % fuzzy exponent  1.1:0.1:2 
dim = 20;    % reduced dimension  k/3 - k 

%% UDFSRSP
[feature_id,W,obj] = SPDFS(fea',c,phi,k,dim);            
X_new = fea(:,feature_id);
lab = litekmeans(X_new,c,'Replicates',1);
result = ClusteringMeasure(gnd,lab); 

fprintf('ACC=%f NMI=%f\n ',result(1),result(2));       


