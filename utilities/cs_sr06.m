% cs_sr06 - Rekonstrukcija signala kod sazetog ocitavanja. 
%
% Poziva se:
%    [xr,Status] = cs_sr06(yk,B);
%    [xr,Status] = cs_sr06(yk,B,Ispis);
% gdje je
%    yk     - vektor dobiven sazetim ocitavanjem (M komponenata)
%    B      - matrica koja povezuje sparse bazu i ocitani vektor (M x N)
%    Ispis  - ako je 1, ispisuju se poruke koje daje SeDuMi
%    xr     - rekonstruirani signal u sparse domeni (N komponenata)
%    Status - sadrzi info koji daje SeDuMi
%
% Algoritam
%    Funkcija rjesava problem rekonstrukcije kod sazetog ocitavanja, oblika 
%       xr = arg min ||x||
%                 x
%       s.t. yk = B*x
%    pri cemu ||.|| oznacava L1 normu.
%    Optimizacijski problem rjesava se pomocu funkcije SeDuMi.
%    Formulacija problema opisana je u knjizi 
%       Irina Rish, Genady Ya. Grabarnik,
%       Sparse Modeling - Theory, Algorithms, and Applications,
%       CRC Press, 2015
%       str. 22-23
%
% Primjeri poziva
%
% Napomene:
%   1. Koristi se sparse prikaz matrica.
%   2. Funkcija cs_sr02 koristi jednakosti i nejednakosti za formulaciju L1 problema.
%      To je zastarjelo. Odsad treba korisitit ovu funkciju.
%   3. Koristan podatak je Status.numerr. Ako je 0, optimizacija je uspjesno provedena.
%      Za detalje vidi help sedumi.
%


function [xr,Status] = cs_sr06(yk,B,Ispis)

% ======================================================================================
%                                  OBRADA ULAZA
% ======================================================================================

if nargin==2
   SedumiParametri.fid=0;
else
   SedumiParametri.fid=Ispis;
end

% ======================================================================================
%                                   REKONSTRUKCIJA
% ======================================================================================

% Polazni problem
%       xr = arg min ||x||
%                 x
%       s.t. yk = B*x         (size(B)=[M,N])
%
% ekvivalentan je problemu    (Standard primal form)
%       zr = arg min z1+z2+...+z2N
%                 z
%       s.t. yk = [B,-B]*z
%            z >= 0
% uz
%       xr = z(1:N) - z(N+1:2*N) 

[M,N] = size(B);

if M >= 2*N

   if SedumiParametri.fid==1;
      fprintf('\n!!! Funkcija cs_sr06: M>=2N (Umjetno regulariziran slucaj)\n\n')
   end
% 
%    c = sparse([ones(2*N,1);zeros(M-2*N+1,1)]);
%    A = [sparse([B,-B]), zeros(M,M-2*N+1)];
%    b = sparse(yk);
   
   c = full([ones(2*N,1);zeros(M-2*N+1,1)]);
   A = [full([B,-B]), zeros(M,M-2*N+1)];
   b = full(yk);
   
   K.l = M+1;
   K.f = 0;         

   [z,z_tmp,Status] = sedumi(A,b,c,K,SedumiParametri);

   zr = z(1:2*N);
   xr = zr(1:N) - zr(N+1:2*N);
   xr=xr(:);

else

   c = full(ones(2*N,1));
   A = full([B,-B]);
   b = full(yk);
   
   K.l = 2*N;
   K.f = 0;
   
   [z,z_tmp,Status] = sedumi(A,b,c,K,SedumiParametri);
   
   xr = z(1:N) - z(N+1:2*N);
   xr=xr(:);

end

