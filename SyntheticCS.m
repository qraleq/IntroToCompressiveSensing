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

addpath('utilities', 'data')
% addpath('/home/ivan/nfs/Projects/MATLAB Projects/Matlab Toolboxes/SpaRSA_2.0')

% define measurement and transformation matrix type
psiType='dwt';

if(strcmp(psiType, 'dwt'))
    waveletType='haar';
end

phiType='spike';

% define optimization package
optimizationPackage = 'sedumi'; % Options: 'cvx', 'sedumi', 'yalmip', 'sparsa', 'fpc', 'cls','sparsaTV'
epsilon = 3;

% choose block size
block_size = 8;

% choose number of measurements used in reconstruction
noOfMeasurements  = 35;     % desired M << N

% set desired sparsity percentage
sparsityPercentage = 0.99999;  % percentage of coefficients to be left

%% CREATE MEASUREMENT AND TRANSFORMATION MATRIX
% In the CS problem, linear bases are usually defined by matrices Phi and Psi

% generate transformation matrix to use in reconstruction process
[psi, psi_inv] = generateTransformationMatrix(psiType, block_size, 'waveletType', waveletType);

[psi, no]= wmpdictionary(256);
% psi_inv = psi(:, 3*255:3*255+255)';

psi = wmpdictionary(block_size^2, 'lstcpt', {{'sym4',5}});

mpdict = wmpdictionary(block_size^2 ,'lstcpt',{'dct','RnIdent', 'sin', 'poly', {'sym4',5}, {'sym4',2}, {'sym4',3}, {'sym4',7},{'wpsym4',5}, {'wpsym4',10} , {'wpsym4',8}});
% mpdict = wmpdictionary(block_size^2 ,'lstcpt',{{'sym4',5}});

psi_inv = mpdict;
% imagesc(abs(mpdict(:,1:256)*mpdict(:,1+256:256+256)'))




% [psi2, psi_inv2] = generateTransformationMatrix('dct', block_size);

% psi_inv = [psi_inv, psi_inv2];

% psi = psi(1:256, 1:256);
% psi_inv = psi_inv(1:256, 1:256);

% phi = dctmtx(64);
% psi = eye(64)';
% psi_inv = eye(64);


% phi = hadamard(block_size^2);
% psi_inv = psi';


% generate measurement matrix to use in measurement process
% phi = generateMeasurementMatrix(phiType, block_size);

% phi = hadamard(block_size^2);

% phi = generateTransformationMatrix('dct', block_size);
% v = double(randn(1,block_size.^2)>0.5);
% phi = toeplitz([v(1) fliplr(v(2:end))], v);


% phi = rand(20,64)>0.5;
% temp = dctmtx(64);
%
% phi = [phi; temp(1:44,:)];
% phi = repmat(phi, [4,4]);

% sigma defines variation in succesive measurement matrices
% this simulates case in which imperfect measurement system is used and in
% that case there is variation in measurement masks
% sigma = 0.007;

% randomVector = ones(block_size^2,1)+(sigma*(rand(block_size^2,1)-sigma));
% randomMatrix = repmat(randomVector, [1,block_size^2]);

% add error to different measurement masks
% phi2 = randomVector.*phi;


% phi2 = imrotate(circshift(phi, [0 randi([9 15], 1)]), 0.1, 'crop');
% phi2 = imrotate(phi, 1, 'crop');

% phi2 = phi;
% phi_a = randomVector.*ones(block_size^2, block_size^2);

% Check coherence
% disp('Coherence between measurement and transformation matrix is:')

% npsi=sqrt(sum(psi.*conj(psi),1));
% nphi=sqrt(sum(phi.*conj(phi),1));

% nMatPsi = bsxfun(@rdivide, psi, npsi);
% nMatPhi = bsxfun(@rdivide, phi, nphi);

% from 1 - incoherent to sqrt(size(phi,2)) - coherent
% coherence = sqrt(size(phi,2))* max(max(abs(nMatPhi*nMatPsi')));

% coherence = max(max(abs(phi*psi)))

% coherence = max(max((abs(corr(phi, psi)))))

% sqrt(size(phi,2)) * max(max(abs(nMatPhi*nMatPsi')))  % From 1 - incoherent to sqrt(size(R,2)) - coherent

% load or create a test image for compressive imaging reconstruction
image = imresize(im2double(rgb2gray(imread('lenna.tiff'))), 0.5);
% image = phantom;
% image = randn(128, 128)>0.7;

% image=imnoise(image, 'salt & pepper', 0.01);
% image=imnoise(image, 'gauss', 0.05);

[rows, cols]=size(image);

figure
subplot(121), imshow(image, 'InitialMagnification', 'fit'), title('Original image'), colormap gray, axis image
image=sparsifyImage(image,[], sparsityPercentage);
subplot(122), imshow(image, 'InitialMagnification', 'fit'), title('Original image - resized and sparsified'), colormap gray, axis image

ind = logical(randerr(1,block_size^2,noOfMeasurements));


%%
tic

% initialize matrix for reconstructed image
image_est = [];

% percentage of used measurements or number of used measurements
% select only M observations out of total N

for k=1:block_size:rows-block_size+1
    for l=1:block_size:cols-block_size+1
        
%                 phi = generateMeasurementMatrix(phiType, block_size);
 
        im1 = image(k:k+block_size-1, l:l+block_size-1);
        
%         im1 = fliplr(flipud(im1));
        
        %         im2=image2(k:k+block_size-1, l:l+block_size-1);
        
        % simulated observation/measurement
        % observation matrix  x  input data converted to 1D
%         y = phi * reshape(im1, block_size*block_size, 1);
                phi = (randn(block_size, block_size) > 0.1);

%         y = reshape(phi(:, 1), [block_size, block_size]) .* im1;
        
%         phi = ones(block_size, block_size);
%         phi(6, 6) = 0;
        
        y = phi .* im1;
% 
        phi_diag = sparse(diag(phi(:)));
        phi_r = phi_diag(any(phi_diag,2),:);
        
%         y = y + rand*1.2;
        

        
%         phi2 = imrotate(circshift(phi2, [0 randi([7 8], 1)]), 0.1, 'crop');
        
%         phi2=phi;
%         phi2(:, 1:5)=0;
%         phi2(:, end-5:end)=0;
        
        %         y_a = (phi_a * reshape(im2(1:block_size,1:block_size), block_size*block_size, 1));
        
        % estimate difference in measurement result caused by noise in
        % measurement matrix rows(measurement masks)
        
        %         corr = (y_a./mean(y_a));
        
        %         corr = (y_a./y_a(1));
        %                 corr = (y_a./32);
        
        %         corr = y_a;
        
        %         y = y./corr;
        
        %         % add noise to measurements or measurement matrix
        %         y_orig = y;
        %         y = y + rand(block_size^2,1)*0.01;
        %         noise_est=abs(y-y_orig);
        %         phi=phi+rand(size(phi, 1), size(phi, 2))/100;
        %         epsilon = 1.3*max(max(std(y)));
        
        % select only M observations out of total N
%         y_m = y(ind);

         y_m = nonzeros(y(:));
            
        % reduced observation matrix (phi_r)
%         phi_r = phi(ind, :);
        
        % define compressive sensing design matrix theta
        theta = phi_r * psi_inv;
        [M,N] = size(theta);
        
        % CS reconstruction - L1 optimization problem
        % min_L1 subject to y_m = Phi_m * Psi^(-1) * s
        
        switch optimizationPackage
            case 'sedumi'
                
                s_est = cs_sr06(y_m, theta);                                
                signal_est = (psi_inv * s_est);
                image_est(k:k+block_size-1, l:l+block_size-1) = reshape(signal_est, [block_size block_size]);
                
                
%                 image_est(k:k+block_size-1, l:l+block_size-1) = reshape(s_est, [block_size block_size]);

%                 s_est_flip = flip(s_est);
%                 signal_est_flip = (psi_inv * s_est_flip);
%                 y_flip = phi_r * signal_est_flip;
%                 
%                 s_est_flip_new = cs_sr06(y_flip, theta);
%                 
%                 s_est_new = flip(s_est_flip_new);
%                 signal_est_new = (psi_inv * s_est_new);
%                 
%                 image_est_new(k:k+block_size-1, l:l+block_size-1) = reshape(signal_est_new, [block_size block_size]);

                
                
%                 image_est(k:k+block_size-1, l:l+block_size-1) = reshape(flipud(fliplr(signal_est)), [block_size block_size]);

                



                
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
                    y_m = y_m + b_Axk;
                    
                end
                
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

                
            case 'fpc'
                for i = 1:1
                    
                    tau = 0;
                    opts.gtol = 1e-8;
                    %                     psi_function = @(x,tau) wthresh(x,'s',tau);
                    %                     phi_function = @(x) norm(x, 1);
                    
                    %                     options.tv_norm='l2';
                    %                     phi_function = @(x) compute_total_variation(x, options);
                    
                    s_est = FPC_AS(64, theta, y_m, tau);
                    
                    signal_est = (psi_inv * s_est).';
                    
                    image_est(k:k+block_size-1, l:l+block_size-1)= reshape(signal_est,[block_size block_size]);
                    
                    % --- ym_k update ---
                    b_Axk = y_m - theta*s_est;
                    
                    test1(:,i)=b_Axk;
                    
                    
                    if(norm(b_Axk) < 1e-5)
                        disp('kraj');
                        break;
                    end
                    
                    y_m = y_m + b_Axk;
                    
                end
                
            case 'cls'
                
                opts = optimset('LargeScale', 'on', 'UseParallel', 1)
                
                [s_est,RESNORM,RESIDUAL,EXITFLAG]=lsqlin(ones(64,64), zeros(64,1), [], [], theta, y_m, [], [], [],  opts);
                
                EXITFLAG
                
                signal_est = (psi_inv * s_est).';
                
                image_est(k:k+block_size-1, l:l+block_size-1)= reshape(signal_est,[block_size block_size]);
                
            case 'sparsaTV'
                
                % reconstruction method using Bregman iterations and SpaRSA
                
                for i = 1:4
                    % noise estimation variable
                    
                    A = @(x) phi_r*x;
                    At = @(x) phi_r'*x;
                    
                    %                     A = @(x) theta*x;
                    %                     At = @(x) theta'*x;
                    
                    % denoising function
                    tv_iters = 3;
                    %                     Psi = @(x,th) tvdenoise(x, 2/th ,tv_iters);
                    %                     Psi = @(x,th) soft(x,th);
                    Psi = @(x,th) tvdenoise_sitcm(x,2/th,tv_iters);
                    
                    % regularizer
                    %                     Phi = @(x) TVnorm_sitcm(x);
                    Phi = @(x) l0norm(x);
                    %                     Phi =@(x) norm(x, 1);
                    
                    %  regularization parameter
                    epsilon = 0.01;
                    
                    % stopping theshold
                    tolA = 1e-10;
                    
                    [s_est, ~, obj_twist,...
                        times_twist, ~, mse_twist]= ...
                        SpaRSA(y_m, A, epsilon,...
                        'AT', At, ...
                        'Phi', Phi, ...
                        'Psi', Psi, ...
                        'Monotone',0,...
                        'MaxiterA', 100, ...
                        'Initialization',0,...
                        'StopCriterion', 2,...
                        'ToleranceA', tolA,...
                        'Verbose', 0);
                    
                    signal_est=s_est;
                    %                     signal_est = (psi_inv * s_est).';
                    
                    image_est(k:k+block_size-1, l:l+block_size-1)= reshape(signal_est,[block_size block_size]);
                    
                    
                    % --- ym_k update ---
                    b_Axk = y_m - theta*s_est;
                    
                    test1(:,i)=b_Axk;
                    
                    if(norm(b_Axk) < 1e-10)
                        disp('kraj');
                        break;
                    end
                    
                    y_m = y_m + b_Axk;
                    
                end
        end
        
        
        
        figure(100)
        %                 imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
        imagesc(image_est), title('Image Reconstruction'), colormap gray, axis image
        drawnow
        
        
%                 figure(101)
%         %                 imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
%         imagesc(image_est_new), title('Image Reconstruction'), colormap gray, axis image
%         drawnow
%         
    end
end

toc

%% VISUALIZATIONS
figure, imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction - final'), colormap gray, axis image

fun = @(block_struct) mean2(block_struct.data);
measurement_visualization = blockproc(image,[block_size block_size],fun);

measurement_visualization=(imresize(measurement_visualization,block_size,'nearest'));
figure
imshow(measurement_visualization, 'InitialMagnification', 'fit'), colormap gray, axis image, title('Measurement')

