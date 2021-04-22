
function [opt_index,W,obj] = SPDFS(X,c,phi,k,m)
% Input
% X: dim*num data matrix, each column is a data point
% c: no. classes
% phi: fuzzy exponent (> 1) for the partition matrix
% k: number of selected features
% m: reduced dimension

% Output
% opt_index: indices of selected features
% W: dim*m projection matrix, which has k nonzero rows
% obj: objective function value

[dim,n] = size(X);
NITER = 20;

%% ===================== Initialization =====================

% initialize U
U = initfcm(c, n);
[U,~,~] = stepfcm(X',U,c,phi);

% initialize W
W = orth(rand(dim,m));

% calculate satter matrix
I = eye(n);
e = ones(n,1);
H = I-(e*e')/n;
St = X*H*X';

F = U'.^phi;
Sm = X*(diag(F*ones(c,1))-F*diag((F'*ones(n,1)).^-1)*F')*X';

%% =====================  updating =====================
obj = [];
for iter = 1:NITER   
    
    % determine lambda
    lambda = trace(W'*Sm*W)/trace(W'*St*W);
    
    obj = [obj;lambda];
    if iter>2 && abs(obj(iter-1)-obj(iter))<10^-4
       break;
    end
    
    % calculate A
    [~,eigvalue,~] = eig1(Sm-lambda*St,1);
    A = eigvalue*eye(dim)-(Sm-lambda*St);
    
    % update 
    [opt_index,W,U,Sm] = step_SPDFS(X,c,phi,k,n,lambda,A,W,U,Sm,St);
    
end

% plot(obj);

end




