
function [opt_index,W,U,Sm] = step_SPDFS(X,c,phi,k,n,lambda,A,W0,U0,Sm,St)
% Input
% X: dim*num data matrix, each column is a data point
% c: no. classes
% phi: fuzzy exponent (> 1) for the partition matrix
% k: number of selected features
% n: sample size
% lambda: paramter corresponding to alpha in this paper
% A: matrix corresponding to Sd in this paper
% W0: W from the last iteration
% U0: U from the last iteration
% Sm: Sm from the last iteration
% St: total scatter matrix

% Output
% opt_index: indices of selected features
% W: dim*m projection matrix, which has k nonzero rows
% U: updated U
% Sm: updated Sm

NITER = 20;

%% =====================  updating =====================

for iter = 1:NITER   
    
    obj_sub(iter) = trace(W0'*Sm*W0)-lambda*trace(W0'*St*W0);
    if iter>2 && abs(obj_sub(iter-1)-obj_sub(iter))<10^-4
       break;
    end
    
    % update W
    [opt_index,W] = L20sparse(A,k,W0);
    W0 = W;
    
    % update U
    data = W'*X;
    [U,~,~] = stepfcm(data',U0,c,phi);
    U0 = U;
    
    % update scatter matrix
    F = U'.^phi;
    Sm = X*(diag(F*ones(c,1))-F*diag((F'*ones(n,1)).^-1)*F')*X';
    
end

% plot(obj_sub);

end




