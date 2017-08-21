%% D. Sersic, M. Vucic, October 2015
%% Update: I. Ralasic, May 2016
% Compressive sensing (CS) - 2D example
%
% Assumption of sparsity
% There is a linear base Psi in which observed phenomenon is sparse:
% only K non-zero spectrum samples out of total N are non-zero, K << N
%
% Examples: Psi == DWT (discrete wavelet transform), DCT (discrete cosine transform),...
%           Many  natural phenomena have sparse DWT, DCT, ... spectra
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
% close all
clc

addpath('utilities', 'data')
% addpath('/home/ivan/nfs/Projects/MATLAB Projects/Matlab Toolboxes/SpaRSA_2.0')

% define measurement(phi) and transformation(psi) matrix type
psiType = 'dwt';

if(strcmp(psiType, 'dwt'))
    waveletType = 'haar';
end
 
phiType = 'spike';

% choose optimization package
optimizationPackage = 'sparsa'; % Options: 'cvx', 'sedumi', 'yalmip', 'sparsa', 'fpc', 'cls','sparsaTV'
epsilon = 0;

% choose block size - divide whole image in non-overlapping blocks
blockSize = 8;

% choose number of measurements used in reconstruction
noOfMeasurements  = 64;     % desired M << N

% set desired sparsity percentage
sparsityPercentage = 0.99;  % percentage of coefficients to be left

%% CREATE MEASUREMENT AND TRANSFORMATION MATRIX
% In the CS problem, linear bases are usually defined by matrices Phi and Psi

% generate transformation matrix to use in reconstruction process
% [psi, psi_inv] = generateTransformationMatrix(psiType, blockSize, 'waveletType', waveletType);
% [psi, psi_inv] = generateTransformationMatrix('dct', blockSize);

noOfWaveletLevels = 2;
% psi_inv = wmpdictionary(blockSize, 'lstcpt', {{'haar', noOfWaveletLevels}});


psi_inv = wmpdictionary(blockSize, 'lstcpt', {'dct'});

% psi_inv = dctmtx(blockSize^2)';
psi_inv = kron(full(psi_inv), full(psi_inv));


% vec_ones = ones(blockSize^2,1)
% mat_ones = diag(vec_ones,1);
% % mat_ones = diag(vec_ones)+mat_ones(1:end-1,1:end-1);
% psi_inv = dct(mat_ones(1:end,1:end))';

% psi_inv = dct(eye(blockSize^2))';

% overcomplete dictionary creation
% psi_inv = wmpdictionary(blockSize^2 ,'lstcpt',{'dct','RnIdent', 'sin', 'poly', {'sym4',5}, {'sym4',2}, {'sym4',3}, {'sym4',7},{'wpsym4',5}, {'wpsym4',10} , {'wpsym4',8}, ...
%     {'haar', 2}, {'haar', 3}, {'haar', 4}, {'haar', 5}, {'haar', 6}, {'haar', 7}, {'haar', 8}, {'haar', 9}});
% psi_inv = wmpdictionary(blockSize^2 ,'lstcpt',{{'haar', 2}, {'haar', 3}, {'haar', 4}, {'haar', 5}, {'haar', 6}, {'haar', 7}, {'haar', 8}, {'haar', 9}});

% mpdict = wmpdictionary(blockSize^2 ,'lstcpt',{{'sym4',5}});

% psi_inv = mpdict;
% imagesc(abs(mpdict(:,1:256)*mpdict(:,1+256:256+256)'))



% [psi2, psi_inv2] = generateTransformationMatrix('dct', blockSize);

% psi_inv = [psi_inv, psi_inv2];

% psi = psi(1:256, 1:256);
% psi_inv = psi_inv(1:256, 1:256);

% phi = dctmtx(64);
% psi = eye(64)';
% psi_inv = eye(64);

% % phi = mtxdya(blockSize^2);
% phi = walsh(blockSize^2);
% phi(phi<0) = 0;
% phi = normalizeColumns(phi);

% phi(1,:) = ~phi(end,:);
% phi(:,1) = ~phi(:, end);

% phi(1:2:end, 1) = 0;
% phi(1,:)=not(phi(2,:));

% phi = phi.*2;

% phi = hadamard(blockSize^2)*psi_inv;
% psi_inv = psi';


% generate measurement matrix to use in measurement process
phi = generateMeasurementMatrix('spike', blockSize);
% phi(1:128,:)=~(phi(129:256,:));



% phi = [];
% 
% for i=1:blockSize^2
%     vec = zeros(blockSize^2);
%     vec(1:i)=1;
%     rand_idx = randperm(blockSize^2);
%     vec = vec(rand_idx);
%     
%     phi = [phi; vec];
% end


% phi = hadamard(blockSize^2);

% phi = generateTransformationMatrix('dct', blockSize);
% v = double(randn(1,blockSize.^2)>0.5);
% phi = toeplitz([v(1) fliplr(v(2:end))], v);


% phi = double(rand(64,64)>0.5);

% phi = ones(64, 64);
% psi = zeros(64, 64);

% temp = dctmtx(64);
%
% phi = [phi; temp(1:44,:)];
% phi = repmat(phi, [4,4]);

% sigma defines variation in succesive measurement masks
% this simulates case in which imperfect measurement system is used and in
% that case there is variation in measurement masks
% add multiplicative error into phi
sigma = 0.;

randomVector = ones(blockSize^2,1) + (sigma*(rand(blockSize^2,1))-sigma);

randomMatrix = repmat(sigma*(rand(blockSize^2,1)-0.5), 1, blockSize^2);

% phi2 = randomVector.*phi;


% randomMatrix = rand(blockSize^2, blockSize^2);
phi2 = phi + randomMatrix;


% phi2=walsh(256);

% phi2 = imrotate(circshift(phi, [0 randi([9 15], 1)]), 0.1, 'crop');
% phi2 = imrotate(phi, 1, 'crop');

% phi2 = phi;
% phi_e = randomVector.*ones(blockSize^2, blockSize^2);

% check coherence between measurement and transformation matrix
% disp('Coherence between measurement and transformation matrix is:')
%
% npsi=sqrt(sum(psi.*conj(psi), 1));
% nphi=sqrt(sum(phi.*conj(phi), 1));
% nMatPsi = bsxfun(@rdivide, psi, npsi);
% nMatPhi = bsxfun(@rdivide, phi, nphi);
%
% % from 1 - incoherent to sqrt(size(phi,2)) - coherent
% coherence = sqrt(size(phi,2)) * max(max(abs(nMatPhi'*nMatPsi)))
%
% coherence = max(max(abs(phi'*psi)))
%
% coherence = max(max((abs(corr(phi, psi)))))
%
% sqrt(size(phi,2)) * max(max(abs(nMatPhi*nMatPsi')))  % From 1 - incoherent to sqrt(size(R,2)) - coherent

% load or create a test image for compressive imaging reconstruction
image = imresize(im2double(rgb2gray(imread('lenna.tiff'))), 0.5);
% image = phantom;
% image = randn(128, 128)>0.7;

% add noise to the test image
% image = imnoise(image, 'salt & pepper', 0.01);
% image = imnoise(image, 'gauss', 0.05);

[rows, cols] = size(image);

figure
subplot(121), imshow(image, 'InitialMagnification', 'fit'), title('Original image'), colormap gray, axis image

image = sparsifyImage(image,[], sparsityPercentage);
subplot(122), imshow(image, 'InitialMagnification', 'fit'), title('Original image - sparsified'), colormap gray, axis image


% choose a random subset of noOfMeasurements (M out of total N)
% ind = logical(randerr(1, blockSize^2, noOfMeasurements));
% ind =  1:256;
% indTemp = logical(randerr(1, blockSize, noOfMeasurements/noOfWaveletLevels));
%
% for i = 1 : noOfWaveletLevels
%
%     indTemp = [indTemp, logical(randerr(1, ))];
%
% end
ind = 1:noOfMeasurements;
% %
% for i = 1:blockSize^2
%     figure(1)
%     imagesc(reshape(phi(:,i), [blockSize, blockSize]))
%     waitforbuttonpress
% end
%
%

%%
tic

% initialize matrix for reconstructed image
image_est = [];
image_dct = [];


for k = 1: blockSize : rows - blockSize + 1
    for l = 1: blockSize : cols - blockSize + 1
        
        %                 phi = generateMeasurementMatrix(phiType, blockSize);
        
        im1 = image(k:k+blockSize-1, l:l+blockSize-1);
        
        %         im1 = fliplr(flipud(im1));
        
        %         im2=image2(k:k+blockSize-1, l:l+blockSize-1);
        
        % simulated observation/measurement
        % observation matrix  x  input data converted to 1D
        y = phi2 * reshape(im1, blockSize*blockSize, 1);
        y_1 = ones(size(phi2)) * reshape(im1, blockSize*blockSize, 1);
        
        
%         y = y - mean(y);
        %         y = phi * im1(:);
        
        %         phi = (randn(blockSize, blockSize) > 0.1);
        
        %         y = reshape(phi(:, 1), [blockSize, blockSize]) .* im1;
        
        %         phi = ones(blockSize, blockSize);
        %         phi(6, 6) = 0;
        
        %         y = phi .* im1;
        
%                 y = y + rand*0.8;
%         
        %         phi_diag = sparse(diag(phi(:)));
        %         phi_r = phi_diag(any(phi_diag,2),:);
        
        
        %         phi2 = imrotate(circshift(phi2, [0 randi([7 8], 1)]), 0.1, 'crop');
        
        %         phi2 = phi;
        %         phi2(:, 1:5)=0;
        %         phi2(:, end-5:end)=0;
        
%                 phi_e = randomVector.*repmat(phi(1,:), [blockSize^2, 1]);
%         phi_e = randomVector.*ones(size(phi));

%           phi_e = randomMatrix + ones(size(phi));
          phi_e = randomMatrix + repmat(phi(1,:), blockSize^2, 1);
    
        %         phi_e = ones(size(phi));
        
        y_e = (phi_e * reshape(image(1:blockSize,1:blockSize), blockSize*blockSize, 1));
%                 y_e = (phi_e * reshape(im1, blockSize*blockSize, 1));

%         y_e = (phi_e * reshape(ones(blockSize,blockSize), blockSize*blockSize, 1));

%         y_e = y_e./256;
%         y=y./128;
        
        
        % estimate difference in measurement result caused by noise in
        % measurement matrix rows(measurement masks)
        
%         y = y./(y_e);
        
%         y=y-1;
%         y = y - y_e;

        %         y = y_e./y;
        
%                         corr = (y_e./mean(y_e));
%                         phi = phi
        %
%                         corr = (y_e(1)./y_e);
%                                 corr = (y_e./128);
%                                 corr = (256./y_e);

%                             phi_temp=phi2./((repmat((corr-1),1,size(phi,1))-phi));

% phi2_temp=phi-repmat(y_e,1,size(phi,1));
% phi=phi2_temp;
%                         corr = y_e;
        
%                         y = y./corr;
        
        %         % add noise to measurements or measurement matrix
        %         y_orig = y;
        %         y = y + rand(blockSize^2,1)*0.01;
        %         noise_est=abs(y-y_orig);
%                         phi2=phi+rand(size(phi, 1), size(phi, 2))/10;
        %         epsilon = 1.3*max(max(std(y)));
        
%         y=psi_inv'*y;
        % select only M observations out of total N
%         y_m = 2*y(ind)-y_1(ind);
        y_m = y(ind);
        %          y_m = nonzeros(y(:));
        
%         % reduced observation matrix (phi_r)
%         phi=walsh(blockSize^2);
%         
        phi_r = phi(ind, :);
        
        % define compressive sensing design matrix theta
        theta = phi_r * psi_inv;
        
%         theta = phi_r;
        [M,N] = size(theta);
        
%         y_m = gpuArray(full(y_m));
%         theta = gpuArray(full(theta));
        
        y_cell{((k-1)/blockSize)+1, ((l-1)/blockSize)+1} = y_m;
        
        image_array{((k-1)/blockSize)+1, ((l-1)/blockSize)+1} = im1;
        
        % CS reconstruction - L1 optimization problem
        % min_L1 subject to y_m = Phi_m * Psi^(-1) * s
        
        switch optimizationPackage
            case 'sedumi'
%                 y_m=psi*y_m;
                s_est = cs_sr06(y_m, theta);
                %                 s_est = wmpalg('omp', y ,psi)
                
                signal_est = (psi_inv * s_est);
%                                 signal_est = (psi*s_est);

                %                 signal_est = phi_r' * y_m;
                image_est(k:k+blockSize-1, l:l+blockSize-1) = reshape(signal_est, [blockSize blockSize]);
                
                image_dct(k:k+blockSize-1, l:l+blockSize-1) = dct2(im1);

                %                 image_est(k:k+blockSize-1, l:l+blockSize-1) = reshape(s_est, [blockSize blockSize]);
                
                %                 s_est_flip = flip(s_est);
                %                 signal_est_flip = (psi_inv * s_est_flip);
                %                 y_flip = phi_r * signal_est_flip;
                %
                %                 s_est_flip_new = cs_sr06(y_flip, theta);
                %
                %                 s_est_new = flip(s_est_flip_new);
                %                 signal_est_new = (psi_inv * s_est_new);
                %
                %                 image_est_new(k:k+blockSize-1, l:l+blockSize-1) = reshape(signal_est_new, [blockSize blockSize]);
                
                %                 image_est(k:k+blockSize-1, l:l+blockSize-1) = reshape(flipud(fliplr(signal_est)), [blockSize blockSize]);
                
            case 'omp'
%                 s_est = sparseCode(y_m, theta, 10, 500);

                s_est = OMP(theta, y_m, 100);
                signal_est = (psi_inv * s_est);
                image_est(k:k+blockSize-1, l:l+blockSize-1) = reshape(signal_est, [blockSize blockSize]);

                
            case 'mex'
%                   y_m=psi_inv'*y_m;

%                 theta=theta./repmat(sqrt(sum(theta.^2)),[size(theta,1) 1]);
                %                 param.pos=true;
%                 param.lambda2=0.01
%                 param.ols=true;
                param.mode=3;
%                 param.pos=1;
                param.cholesky=true;
                param.lambda=0.1; % not more than 20 non-zeros coefficients
%                 param.L=2; % not more than 10 non-zeros coefficients
                param.eps=0; % squared norm of the residual should be less than 0.1
                param.numThreads=-1; % number of processors/cores to use; the default choice is -1
                % and uses all the cores of the machine
%                 W=rand(size(theta,2),1);
%                 W=1:size(theta,2);
%                 W=size(theta,2):-1:1;
%                 W=ones(size(theta,2),1);
%                 W=W';

%                 s_est = mexOMP(y_m, theta, param);
                s_est = mexLasso(y_m, theta, param);
                %                 s_est = wmpalg('omp', y ,psi)
                
                signal_est = (psi_inv * s_est);
%                                 signal_est = (s_est);

                %                 signal_est = phi_r' * y_m;
                image_est(k:k+blockSize-1, l:l+blockSize-1) = reshape(signal_est, [blockSize blockSize]);
                
%                 image_dct(k:k+blockSize-1, l:l+blockSize-1) = dct2(im1);
                
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
                    
%                     if(strcmp(psiType,'dct'))
                        signal_est = (psi_inv * s_est).';
%                         
%                     elseif(strcmp(psiType,'dwt'))
%                         signal_est = waverec2(s_est, S, waveletType); % wavelet reconstruction (inverse transform)
%                     end
                    
                    image_est(k:k+blockSize-1, l:l+blockSize-1)= reshape(signal_est,[blockSize blockSize]);
%                     image_sparse(k:k+blockSize-1, l:l+blockSize-1)=im;
                    

                    
%                     % --- bk update ---
%                     b_Axk = y_m - theta*s_est;
%                     test1(:,i)=b_Axk;
%                     if norm(b_Axk) < 1e-14; break; end
%                     y_m = y_m + b_Axk;
                    
                end
                
            case 'yalmip'
                
                s_est=sdpvar(N,1);
                %                 solvesdp([theta*s_est == y_m], norm(s_est,1));
                solvesdp([norm(theta*s_est-y_m) <= epsilon], norm(s_est,1));
                
                signal_est = (psi_inv * s_est).';
                %                 signal_est=double(s_est);
                
                image_est(k:k+blockSize-1, l:l+blockSize-1)= reshape(signal_est,[blockSize blockSize]);
                
                figure(100)
                imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
                drawnow
                
            case 'sparsa'
                % reconstruction method using Bregman iterations and SpaRSA
                tau = 1e-10;
                tolA = 1e-20;

                [s_est, ~, objective] = SpaRSA(y_m, theta, tau,...
                    'Debias',0,...
                    'Initialization',0,...
                    'StopCriterion', 0,...
                    'ToleranceA', tolA,...
                    'Verbose', 0,...
                    'Continuation', 0);
                
                                    
                signal_est = (psi_inv * s_est);
                
                image_est(k:k+blockSize-1, l:l+blockSize-1)= reshape(signal_est,[blockSize blockSize]);
                
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
                    
                    image_est(k:k+blockSize-1, l:l+blockSize-1)= reshape(signal_est,[blockSize blockSize]);
                    
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
                
                image_est(k:k+blockSize-1, l:l+blockSize-1)= reshape(signal_est,[blockSize blockSize]);
                
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
                    
                    image_est(k:k+blockSize-1, l:l+blockSize-1)= reshape(signal_est,[blockSize blockSize]);
                    
                    
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
        
        
%             figure(100)
%     %                 imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
%     imagesc(image_est), title('Image Reconstruction'), colormap gray, axis image
%     drawnow

        
        
        %                 figure(101)
        %         %                 imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
        %         imagesc(image_est_new), title('Image Reconstruction'), colormap gray, axis image
        %         drawnow
        %
    end
    figure(100)
%                     imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction'), colormap gray, axis image
    imagesc(image_est), title('Image Reconstruction'), colormap gray, axis image
    drawnow
end

toc

%% VISUALIZATIONS

figure, imshow(image_est, 'InitialMagnification', 'fit'), title('Image Reconstruction - final'), colormap gray, axis image

fun = @(block_struct) mean2(block_struct.data);
measurement_visualization = blockproc(image,[blockSize blockSize],fun);

measurement_visualization=(imresize(measurement_visualization,blockSize,'nearest'));
figure
imshow(measurement_visualization, 'InitialMagnification', 'fit'), colormap gray, axis image, title('Measurement')


% 
% 
% %% SBS-LBR
% 
% blockSize_lbr = 4;
% 
% phi_lbr = kron(eye(blockSize_lbr), phi_r);
% % psi_inv_lbr = kron(eye(blockSize_lbr), psi_inv);
% psi_inv_lbr = dctmtx(blockSize_lbr*blockSize^2)';
% 
% theta = phi_lbr * psi_inv_lbr;
% 
% for i=1:size(y_cell,1)-1
%     for j=1:size(y_cell,2)-1
%         
%         i,j
%         
%         y_reconstruct = y_cell(i:i+1, j:j+1);
%         y_reconstruct = y_reconstruct(:);
%         y_reconstruct = cell2mat(y_reconstruct);
% %         
% %         
% %         s_est = cs_sr06(y_reconstruct, theta);
% %         signal_est = (psi_inv_lbr * s_est);
% %         
% %         image_est_tmp = reshape(signal_est, [blockSize^2, blockSize_lbr]);
% %         
% %         image_est_lbr{i,j} = col2im(image_est_tmp, [blockSize, blockSize], [2*blockSize, 2*blockSize], 'distinct');
% %         image_est_lbr{i,j} = mat2cell(image_est_lbr{i,j}, [blockSize, blockSize], [blockSize, blockSize]);
% %         
% %         
%         
%         param.mode=2;
%         param.cholesky=true;
%         param.lambda=0.001; % not more than 20 non-zeros coefficients
%         %                 param.L=500; % not more than 10 non-zeros coefficients
%         param.eps=0; % squared norm of the residual should be less than 0.1
%         param.numThreads=-1; % number of processors/cores to use; the default choice is -1
%         % and uses all the cores of the machine
%                         W=rand(size(theta,2),1)';
% %         W=1:size(theta,2);
%         %                 W=size(theta,2):-1:1;
%         %                 W=ones(size(theta,2),1)';
%         W=W';
%         
%         s_est = mexLassoWeighted(y_reconstruct, theta, W, param);
%         %                 s_est = wmpalg('omp', y ,psi)
%         
%         signal_est = (psi_inv_lbr * s_est);
%         
%         image_est_tmp = reshape(signal_est, [blockSize^2, blockSize_lbr]);
%         
%         image_est_lbr{i,j} = col2im(image_est_tmp, [blockSize, blockSize], [2*blockSize, 2*blockSize], 'distinct');
%         image_est_lbr{i,j} = mat2cell(image_est_lbr{i,j}, [blockSize, blockSize], [blockSize, blockSize]);
%         % figure
%         % imagesc(cell2mat(image_est{i,j}))
%         
% %         if(i==size(y_cell,1) && j==size(y_cell,2))
% %             y_reconstruct = [y_cell(i, j), zeros(size(blockSize)); zeros(size(blockSize)), zeros(size(blockSize))];
% % 
% %         end
%         
%     end
% end
% 
% 
% for i=1:size(y_cell,1)-1
%     for j=1:size(y_cell,2)-1
%         
%         image_lbr{i,j} = [(image_est_lbr{i,j}{1,1})];
%         
%     end
% end
% 
% 
% figure, colormap gray
% imagesc(cell2mat(image_lbr)), axis image
% 
% %%
% 
% for i=1:size(y_cell,1)-2
%     for j=1:size(y_cell,2)-2
%         
%         
%         %         figure(1)
%         %         imagesc(cell2mat((image_est_lbr{i,j})))
%         %         drawnow
%         
%         test{i,j} = cell2mat((image_est_lbr{i,j}));
%         
% %         image_lbr_weighted{i,j} = ((image_est_lbr{i,j}{2,2}) + (image_est_lbr{i,j+1}{2,1}) + (image_est_lbr{i+1,j}{1,2}) + (image_est_lbr{i+1,j+1}{1,1}))/4;
%         image_lbr{i+1,j+1} = ((image_est_lbr{i,j}{2,2}) + (image_est_lbr{i,j+1}{2,1}) + (image_est_lbr{i+1,j}{1,2}) + (image_est_lbr{i+1,j+1}{1,1}))/4;
%         
%         
%         
%         % imagesc(cell2mat(test))
%         
%         %%
%         
%     end
% end
% 
% figure, colormap gray
% imagesc(cell2mat(image_lbr)), axis image
% 
% 


%%

blockSize = 8;
% psi_inv = wmpdictionary(8, 'lstcpt', {{'haar', noOfWaveletLevels}});
% psi_inv = wmpdictionary(blockSize^2, 'lstcpt', {'dct'});
phi = mtxdya(blockSize^2);
% phi=eye(blockSize^2);
% phi = phi(1:end,:)
% phi = walsh(blockSize^2);
psi_inv = wmpdictionary(blockSize, 'lstcpt', {{'haar', 5}});

% psi_inv = dctmtx(blockSize^2)';
psi_inv = kron(full(psi_inv), full(psi_inv));

theta = phi*psi_inv;

% [~,I] = sort(sum(theta~=0,2), 'descend');
% theta = theta(I,:);
% theta = sortrows()
% imagesc(psi_inv)
imagesc(theta)
%%
% n=64
% a=1/sqrt(n);
% for i=1:n
%     H(1,i)=a;
% end
% for k=1:n-1
%     p=fix(log2(k));
%     q=k-2^p+1;
%     t1=n/2^p;
%     sup=fix(q*t1);
%     mid=fix(sup-t1/2);
%     inf=fix(sup-t1);
%     t2=2^(p/2)*a;
%     for j=1:inf
%         H(k+1,j)=0;
%     end
%     for j=inf+1:mid
%         H(k+1,j)=t2;
%     end
%     for j=mid+1:sup
%         H(k+1,j)=-t2;
%     end
%     for j=sup+1:n
%         H(k+1,j)=0;
%     end
% end
% 
% %%
% 
% I = imread('lenna.tiff');
% [M N] = size(I);
% if M ~= N
% end
% H = haar_basis(N);
% P = permutation_matrix(N);
% A = P*H;
% B = H'*P';
% J = A*double(I)*B; % forward transform
% % haar_basis
% %
% function H = haar_basis(N)
% haar_wavelet = [1 1; 1-1];
% H = (l/sqrt(2)) .* kron(eye(N/2),haar_wavelet);
% end
% 
