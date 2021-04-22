clc;
close all;

addpath('./'); addpath('./data'); addpath('./fcm'); addpath('./funs'); addpath('./SPDFS');

%% load data
load BASEHOCK.mat
   
%% Preprocessing
fea = X; gnd = Y; c = max(Y);
num = size(fea,1);
fea = normalizefea(num, fea); 

%% Parameter setting
k = 50;      % number of selected features  50:50:300  
phi = 1.2;   % fuzzy exponent  1.1:0.1:2 
dim = 20;    % reduced dimension  k/3 - k    

%% UDFSRSP
% Partition data
[Train_data, Train_label, Test_data, Test_label] = Ransample_vecrate(fea,gnd,0.2);
trfea = Train_data; ttfea = Test_data; trgnd = Train_label; ttgnd = Test_label;  
            
[feature_id,W,obj] = SPDFS(trfea',c,phi,k,dim); 
Data_train = trfea(:,feature_id);
Data_test = ttfea(:,feature_id);
                    
knn = 1;
result = Semi_KNN(Data_train,trgnd,Data_test,ttgnd,knn,'Euclidean');

fprintf('Classification Accuracy=%f\n ',result);                           
                    

