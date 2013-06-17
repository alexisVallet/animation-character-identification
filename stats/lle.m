function [Y] = lle(X,K,d)

[D,N] = size(X);

X2 = sum(X.^2,1);
X3 = 2*X'*X;
distance = repmat(X2,N,1)+repmat(X2',1,N)-X3;
[sorted,index] = sort(distance);
neighborhood = index(2:(1+K),:);

if(K>D)  
  tol=1e-3; 
else
  tol=0;
end
W = zeros(K,N);
for ii=1:N
   z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); 
   C = z'*z;                     
   if( rank(C) < K )
       C = C + eye(K,K)*(1e-3)*trace(C);
   else
       C = C + eye(K,K)*tol*trace(C);                 
   end
   W(:,ii) = C\ones(K,1);                           
   W(:,ii) = W(:,ii)/sum(W(:,ii));               
end

M = sparse(1:N,1:N,ones(1,N),N,N,4*K*N); 

for ii=1:N
   w = W(:,ii);
   jj = neighborhood(:,ii);
   M(ii,jj) = M(ii,jj) - w';
   M(jj,ii) = M(jj,ii) - w;
   M(jj,jj) = M(jj,jj) + w*w';
end

options.disp = 0; options.isreal = 1; options.issym = 1; 
[Y,eigenvalues] = eigs(M,d+1,0,options);
Y=fliplr(Y);
Y = Y(:,2:d+1)'*sqrt(N);  
