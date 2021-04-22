function [opt_index,W] = L20sparse(A,k,W0)
% Input
% A: positive semi-definite matrix
% k: number of selected features
% W0: initialization. The input W0 must row-sparse.

% Output
% opt_index: indices of selected features
% W: dim*m projection matrix, which has k nonzero rows


NITER_W = 20;
obj = zeros(NITER_W,1);  
[dim,m] = size(W0);

%% ===================== Initialization =====================

%initialize W
W = W0;

%% =====================  updating =====================

for iter = 1:NITER_W
   P = A*W*pinv(W'*A*W)*W'*A;
   [~, ind] = sort(diag(P), 'descend');
   opt_index = sort(ind(1:k));
   Aopt = A(opt_index, opt_index);
   %[V, ~] = eigs(Aopt, m);
   V = eig1(Aopt, m);
   W = zeros(dim,m);
   W(opt_index, :) = V;
   
   % Since the initialized W0 (in our method) is not row-sparse, the first iteration is 
   % used to make W0 row-sparse. And the obtained W after the first
   % iteration is regarded as the initialized W. That is, the input W of the
   % 'L20sparse' function must row-sparse so that the non-decreasing
   % property can be guaranteed.
   obj(iter) = trace(W'*A*W);
   if iter>2 && abs(obj(iter)-obj(iter-1))<10^-4
      break;
   end 
end

% plot(obj);

end
