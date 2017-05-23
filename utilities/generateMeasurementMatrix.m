function [phi]=generateMeasurementMatrix(phiType, blockSize, varargin)

p=inputParser;

p.addRequired('phiType', @isstr);
p.addRequired('blockSize', @isnumeric);
p.addParameter('Plot', 0, @isnumeric);

p.parse(phiType, blockSize, varargin{:});


if(strcmp(phiType,'gauss'))
    phi = randn(blockSize*blockSize);  
    
elseif(strcmp(phiType,'spike'))
    
    percentage=0.5;
    
    for maskNo=1:blockSize*blockSize
        
        randomMask{maskNo} = zeros(1,blockSize*blockSize) ;
        randomMask{maskNo}(1:round(percentage*blockSize*blockSize)) = 1 ;
        randomMask{maskNo} = randomMask{maskNo}(randperm(numel(randomMask{maskNo})));
        
        if(maskNo==1)
            phi=randomMask{1};
        else
            phi=[phi randomMask{maskNo}];
        end
        
    end
    phi=reshape(phi, blockSize*blockSize, blockSize*blockSize)';
end

if(p.Results.Plot)
    figure, imagesc(phi), colormap gray, title('Measurement Matrix - Phi'),  axis image
end

end