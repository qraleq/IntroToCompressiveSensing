%% D. Sersic, M. Vucic, October 2015
%% Update: I. Ralasic, May 2016
% Compressive sensing (CS) - 2D example
%
% Assumption of sparsity
% There is a linear base Psi in which observed phenomenon is sparse:
% only K non-zero spectrum samples out of total N are non-zero, K << N
%
% Examples: Psi == DWT (discrete wavelet transform), DCT (discrete cosine transform),...
%           Many natural phenomena have sparse DWT, DCT, ... spectra
%
% Assumption of incoherence
% Observation of the phenomenon is conducted in linear base Phi that is incoherent with Psi
%
% CS hypothesis
% The observed phenomenon can be estimated in a statistically reliable way from
% only M >= K  << N observations by solving:
%
% min_L1 s
% subject to y_m = Phi_m * Psi^(-1) * s
%
% Phi_m contains only M rows of observation matrix Phi, where M << N
% (e.g only M observations of y are available)
% s - sparse representation of the observed phenomenon in linear base Psi:
%     observed phenomenon x = Psi^(-1) * s
% y - observations (measurements) in linear base Phi:
%     y = Phi * x

%% WORKSPACE INITIALIZATION
clearvars
close all
clc

addpath('utilities')
addpath('data')
% addpath('/home/ivan/nfs/Projects/MATLAB Projects/Matlab Toolboxes/SpaRSA_2.0')

% define measurement and transformation matrix type
psiType='dct';
waveletType='haar';

phiType='spike';

% Different optimization packages
optimizationPackage = 'sparsa'; % Options: 'cvx', 'sedumi', 'yalmip', 'sparsa', 'fpc'

% Choose block size
block_size = 8;

noOfMeasurements  = 64; % desired M << N
sparsityPercentage = 0.99; % percentage of coefficients to be left

%% CREATE MEASUREMENT AND TRANSFORMATION MATRIX
% In the CS problem, linear bases are usually defined by matrices Phi and Psi

[psi, psi_inv, C, S]=generateTransformationMatrix(psiType,waveletType, block_size);
% phi = generateMeasurementMatrix(phiType, block_size);


phi=hadamard(block_size/2);

% % low frequency capturing measurement matrix
% phi = dct2(phi);
% phi(:, 1:0:end)=0;
% phi=idct2(phi);

% [U, D, V]=svd(phi);

% phi=D;
% 
% n=log2(block_size^2);
% H = hadamard(2^n); % / sqrt(n)
% 
% % H=1;
% % 
% % for i = 1:5
% %     H = [H  H
% %          H -H]; %#ok
% % end
% 
% 
% 
% 
% W = walsh(2^n);
% % W(W<0)=0
% 
% figure, colormap gray
% subplot(121)
% imagesc(H), axis image
% subplot(122)
% imagesc(W), axis image
% 
% % test=horzcat(reshape(W, 4*16, 4), fliplr(reshape(W, 4*16, 4)));
% 
% 
% 
% figure, colormap gray
% imagesc(reshape(W, 8*16, 2)), axis image
% % imagesc(test), axis image
% 
% test=[];
% figure, colormap gray
% 
% for i=1:2^block_size
%    subplot(4,4,i)
%     test=reshape(W(:,i), block_size,block_size)
%    imagesc(test), axis image
% 
%    drawnow
%    waitforbuttonpress
% end
% 
% phi=W*psi;


% Check coherence
disp('Coherence between measurement and transformation matrix is:')

% psi=psi_inv

npsi=sqrt(sum(psi.*conj(psi),1));
nphi=sqrt(sum(phi.*conj(phi),1));

nMatPsi = bsxfun(@rdivide,psi,npsi);
nMatPhi = bsxfun(@rdivide,phi,nphi);

% from 1 - incoherent to sqrt(size(phi,2)) - coherent
coherence = sqrt(size(phi,2))* max(max(abs(nMatPhi*nMatPsi')))  

% coherence = max(max((abs(corr(phi, psi)))))


% sqrt(size(phi,2)) * max(max(abs(nMatPhi*nMatPsi')))  % From 1 - incoherent to sqrt(size(R,2)) - coherent


image = imresize(im2double(rgb2gray(imread('lenna.tiff'))), 0.25);

% image=ones(48, 96);

% image=imnoise(image, 'salt & pepper', 0.01);
% image=imnoise(image, 'gauss', 0.01);

% figure
% imshow(image, 'InitialMagnification', 'fit'), title('Original image'), colormap gray, axis image

[rows, cols]=size(image);

image=sparsifyImage(image,[], sparsityPercentage);

figure
imshow(image, 'InitialMagnification', 'fit'), title('Original image - resized and sparsified'), colormap gray, axis image

%%
tic

image_est = [];

for k=1:block_size:rows-block_size+1
    for l=1:block_size:cols-block_size+1
        
        im=image(k:k+block_size-1, l:l+block_size-1);
        
        
        % Simulated observation
        
        % Observation matrix  x  input data converted to 1D
        % y = R * reshape(im, rows*cols, 1);  % Real data
        y = phi * im * phi'; % Ideally K sparse data
        
        % add noise to measurements or measurement matrix
        %         y_orig = y;
        %         y = y + rand(64,1);
        %         noise_est=abs(y-y_orig);
        %
        %
        %         phi=phi+rand(size(phi, 1), size(phi, 2))/100;
        
        %         epsilon = 1.3*max(max(std(y)));
        
        
        
        % percentage of used measurements or number of used measurements
        % select only M observations out of total N
        ind = logical(randerr(1,block_size^2,noOfMeasurements));
        
        y_m = y(ind);
        y_mk = y_m;
        
        % reduced observation matrix (Phi_m)
        phi_r = phi(ind, :);
        
        
% CS reconstruction - L1 optimization problem
        
        % min_L1 subject to y_m = Phi_m * Psi^(-1) * s
        
        theta = phi_r*psi_inv;
        [M,N]=size(theta);
        
        switch optimizationPackage
            case 'sedumi'
                
                % Standard dual form: data conditioning for minimum L1
                b = [ spalloc(N,1,0); -sparse(ones(N,1)) ];
                
                At = [ -sparse(theta) , spalloc(M,N,0) ;...
                    sparse(theta) , spalloc(M,N,0) ;...
                    speye(N)         , -speye(N)      ;...
                    -speye(N)         , -speye(N)      ;...
                    spalloc(N,N,0)   , -speye(N)      ];
                
                c = [ -sparse(y_mk(:)); sparse(y_mk(:)); spalloc(3*N,1,0) ];
                
                % Optimization
                pars.fid=0; % suppress output
                K.l = max(size(At));
                
                %                 tic
                
                [~,s_est]=sedumi(At,b,c,K,pars); % SeDuMi
                
                %                 toc
                
                % Output data processing
                s_est=s_est(:);
                s_est=s_est(1:N);
                
                if(strcmp(psiType,'dct'))
                    signal_est = (psi_inv * s_est).';
                    
                elseif(strcmp(psiType,'dwt'))
                    signal_est = waverec2(s_est, S, waveletType); % wavelet reconstruction (inverse transform)
                end
                
                
                image_est(k:k+block_size-1, l:l+block_size-1) = reshape(signal_est, [block_size block_size]);
                image_sparse(k:k+block_size-1, l:l+block_size-1)=im;
                
                
                figure(100)
                imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
                drawnow
                
                
            case 'cvx'
                for i=1:1
                    %                 cvx_solver sedumi
                    
                    cvx_begin quiet
                    variable s_est(N, 1);
                    minimize( norm(s_est, 1) );
                    subject to
                                    theta * s_est == y_m;
%                     norm(theta*s_est - y_mk, 2)<=epsilon
                    cvx_end
                    
                    if(strcmp(psiType,'dct'))
                        signal_est = (psi_inv * s_est).';
                        
                    elseif(strcmp(psiType,'dwt'))
                        signal_est = waverec2(s_est, S, waveletType); % wavelet reconstruction (inverse transform)
                    end
                    
                    image_est(k:k+block_size-1, l:l+block_size-1)= reshape(signal_est,[block_size block_size]);
                    image_sparse(k:k+block_size-1, l:l+block_size-1)=im;
                    
                    
                    
                    
                    % --- bk update ---
                    b_Axk = y_m - theta*s_est;
                    test1(:,i)=b_Axk;
                    if norm(b_Axk) < 1e-14; break; end
                    y_mk = y_mk + b_Axk;
                    
                end
                
                figure(100)
                imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
                drawnow
                
                
            case 'yalmip'
                
                s_est=sdpvar(N,1);
                %                 solvesdp([theta*s_est == y_m], norm(s_est,1));
                solvesdp([norm(theta*s_est-y_m) <= epsilon], norm(s_est,1));
                
                signal_est = (psi_inv * s_est).';
                %                 signal_est=double(s_est);
                
                image_est(k:k+block_size-1, l:l+block_size-1)= reshape(signal_est,[block_size block_size]);
                
                figure(100)
                imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
                drawnow
                
                
            case 'sparsa'
                % reconstruction method using Bregman iterations and SpaRSA
                
                for i = 1:1
                    % noise estimation variable
                    tau = 0.03;
                    
                    s_est = SpaRSA(y_mk, theta, tau,...
                        'Debias',0,...
                        'Initialization',0,...
                        'StopCriterion',2,...
                        'ToleranceA',1e-5,...
                        'Verbose', 0,...
                        'Continuation', 0, ...
                        'Monotone', 0);
                    
                    signal_est = (psi_inv * s_est).';
                    
                    image_est(k:k+block_size-1, l:l+block_size-1)= reshape(signal_est,[block_size block_size]);
                    
                    % BREGMAN ITERATION UPDATE
                    b_Axk = y_m - theta*s_est;
         
                    if(norm(b_Axk) < 1e-5)
                        disp('kraj');
                        break;
                    end
                    
                    y_mk = y_mk + b_Axk;
                    
                end
                
                
                figure(100)
                imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
                drawnow
                
                
            case 'fpc'
                for i = 1:1
                    
                    tau = 0;
                    opts.gtol = 1e-8;
                    %                     psi_function = @(x,tau) wthresh(x,'s',tau);
                    %                     phi_function = @(x) norm(x, 1);
                    
                    %                     options.tv_norm='l2';
                    %                     phi_function = @(x) compute_total_variation(x, options);
                    
                    s_est = FPC_AS(64, theta, y_mk, tau);
                    
                    signal_est = (psi_inv * s_est).';
                    
                    image_est(k:k+block_size-1, l:l+block_size-1)= reshape(signal_est,[block_size block_size]);
                    
                    % --- ym_k update ---
                    b_Axk = y_m - theta*s_est;
                    
                    test1(:,i)=b_Axk;
                    
                    
                    if(norm(b_Axk) < 1e-5)
                        disp('kraj');
                        break;
                    end
                    
                    y_mk = y_mk + b_Axk;
                    
                end
                
                
                figure(100)
                imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
                drawnow
        end
    end
end

toc

% VISUALIZATIONS
figure, imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction - final'), colormap gray, axis image

fun = @(block_struct) mean2(block_struct.data);
measurement_visualization = blockproc(image,[block_size block_size],fun);

measurement_visualization=(imresize(measurement_visualization,block_size,'nearest'));
figure
imshow(measurement_visualization, 'InitialMagnification', 'fit'), colormap gray, axis image, title('Measurement')

