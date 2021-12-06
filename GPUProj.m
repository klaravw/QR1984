
% n  = 100;
% s1 = logspace(-1,1,n);
% s2 = logspace(-5,1,n);
% s3 = logspace(-25,1,n);
% U  = orth(rand(n));
% V  = orth(rand(n));
% A1 = U*diag(s1)*V';
% A2 = U*diag(s2)*V';
% A3 = U*diag(s3)*V';
% 
% [q1,r1] = cgs(A1);

A1 = [1 2 3 4; 5 6 7 8; 9 10 10 12; 13 14 15 11]'

 [mq2,mr2]= mgs2(A1)
 [mq3,mr3]= qr(A1,0)
 
 % checking QR properties:
 mq2*mr2
 myMGSrank = rank(mq2)
 myMGSorthonormality = mq2'*mq2
 norm(myMGSorthonormality-eye(4))


function [Qhat,Rhat] = mgs(A)
    [m, n] = size(A);
    r(1,1) = norm(A(:,1));
    q(:,1) = A(:,1)/r(1,1);
    Qhat = q(:,1);
    Rhat = r(1,1);
    % loop over rows
    for j=2:n
        v(:,j) = A(:,j);
        % loop over columns
        for i = 1:j-1
           r(i,j) = q(:,i)'*v(:,j);
           v(:,j) = v(:,j)-q(:,i)*r(i,j);
        end
    r(j,j) = norm(v(:,j));
    q(:,j) = v(:,j)/r(j,j);
    end
    Qhat = q;
    Rhat = r;
end

function [q,r] = mgs2(A)
    q = zeros(size(A)); 
    v = zeros(size(A));
    n = size(A,2);
    r = zeros(n,n);
    for i =1:n
        v(:,i) = A(:,i);
    end
    for i =1:n
        r(i,i) = norm(v(:,i));
        q(:,i) = v(:,i)/r(i,i);
        for j=(i+1):n
            r(i,j)=q(:,i)'*v(:,j);
            v(:,j) = v(:,j)-r(i,j)*q(:,i);
        end
    end
end

function [Qhat,Rhat] = cgs(A)
    [m, n] = size(A);
    r11 = norm(A(:,1));
    q1 = A(:,1)/r11;
    Qhat = q1;
    Rhat = r11;
    for j=2:n
        r = Qhat'*A(:,j);
        v_j = A(:,j)- Qhat*r;
        r_jj = norm(v_j);
        q_j = v_j/r_jj;
        Qhat = [Qhat, q_j];
        Rhat = [Rhat, r; zeros(1,j-1) r_jj];
    end
end