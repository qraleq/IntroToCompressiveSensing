function D = mtxdya(N);
% Generating dyadic(Paley) ordered HADAMARD matrix, 
% N must be a power of 2.
warning off

q = log2(N);

if  sum(ismember(char(cellstr(num2str(q))),'.'))~=0
    disp('           Warning!...               ');
    disp('The size of Vector  must be in the shape of 2^N ..');
    return
else
    
    
for u = 1:N
    binu = dec2bin(u-1,q);
    for v = 1:N
        binv = dec2bin(v-1,q);
        temp = 0;
        for i = 1:q
            temp= temp + bin2dec(binu(i))*bin2dec(binv(q+1-i));
        end
       D(u,v)=(-1)^temp;
    end
end
end