classdef KGGaussian < Kernel
    %KGGAUSSIAN Kernel for distributions defined as the Gaussian kernel on
    %the mean embeddings into Gaussian RKHS.
    %
    % Outer Gaussian kernel with parameter w2. The distributions
    % in X1, X2 are also embedded with a Gaussian kernel with parameter
    % embed_width2
    %
    % Input: X1, X2 = 1xn array of DistNormal.
    %
    % !! THIS ONLY WORKS FOR 1D GAUSSIAN FOR NOW !!
    %
    properties (SetAccess=private)
        % Gaussian kernel for embedding
        kegauss;
        % width2 for mean embeddings. 
        % Generally there should be one parameter for each dimension of the 
        % input distributions. !! DO THIS LATER !!
        embed_width2;
        % width2 for the outer Gaussian kernel on the mean embeddings.
        width2;
        
    end
    
    methods
        function this=KGGaussian(embed_width2, width2)
            % sigm2 = Width for embedding into Gaussian RKHS
            % width2 = Gaussian width^2. Not the one for embedding the
            % distribution.
            assert(width2 > 0, 'Gaussian width must be > 0');
            assert(embed_width2 >0);
            this.kegauss = KEGaussian(embed_width2);
            this.embed_width2 = embed_width2;
            this.width2 = width2;
        end
        
        
        function Kmat = eval(this, data1, data2)
            %             [Kmat, D2] = kerGGaussian(data1, data2, this.sigma2, this.width2);
            Kmat = kerGGaussian(data1, data2, this.embed_width2, this.width2);
        end
        
        
        function Kvec = pairEval(this, X, Y)
            % If X, Y are not Gaussian, we will treat them as one by doing moment 
            % matching i.e., extract mean and variance and construct a Gaussian 
            % out of them.
            assert(isa(X, 'Distribution'));
            assert(isa(Y, 'Distribution'));
            assert(length(X)==length(Y));

            sigma2 = this.embed_width2;
            w2 = this.width2;
            
            if X(1).d==1
                % operation on obj array can be expensive..
                M1 = [X.mean];
                V1 = [X.variance];
                M2 = [Y.mean];
                V2 = [Y.variance];
                
                T1 = KEGauss1.self_inner1d(M1, V1, sigma2);
                T2 = KEGauss1.self_inner1d(M2, V2, sigma2);
                Cross = this.kegauss.pairEval(X, Y);
                Kvec = exp(-(T1-2*Cross+T2)/(2*w2) );
            else
                error('later for multivariate');
            end
        end
        
        function Param = getParam(this)
            Param = {this.embed_width2, this.width2};
        end
        
        function s=shortSummary(this)
            s = sprintf('%s(%.2g, %.2g)', mfilename, this.embed_width2, this.width2 );
        end
    end
    
    methods (Static)
        
        function [DD, M]=compute_meddistances(X, xembed_widths, subsamples)
            % for every embed width, compute the pairwise median distance
            % xembed_widths in a list.
            % High storage cost for DD.
            % - subsamples is a integer denoteing the size of subsamples
            % that will be used instead to compute the meddistance.
            assert(isa(X, 'Distribution'));
            X = DistArray(X);
            if nargin >=3 && subsamples < length(X)
                I = randperm(length(X), subsamples);
                X = X.get(I); %not DistArray anymore
            end
            
            M = zeros(1, length(xembed_widths));
            DD = cell(1, length(xembed_widths));
            for i=1:length(xembed_widths)
                sig2 = xembed_widths(i);
                D2 = distGGaussian( X, X, sig2);
                DD{i} = D2;
                M(i) = median(D2(:));
            end
            
        end
        
        function Kcell = candidates(X, embed_widths, med_factors, subsamples)
            % Generate a cell array of KGGaussian candidates from a list of
            % embeding widths, embed_widths, and a list of factors med_factors 
            % to be  multipled with the pairwise median distance of the mean
            % embeddings.
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.
            
            if nargin < 4
                subsamples = length(X);
            end
            assert(isa(X, 'DistNormal'));
            assert(isnumeric(embed_widths));
            assert(~isempty(embed_widths));
            assert(isnumeric(med_factors));
            assert(~isempty(med_factors));
            assert(all(embed_widths > 0));
            assert(all(med_factors > 0));
            
            Ks = cell(length(embed_widths), length(med_factors));
            for i=1:length(embed_widths)
                ewidth = embed_widths(i);
                [~, med]= KGGaussian.compute_meddistances(X, ewidth, subsamples);
                for j=1:length(med_factors)
                    fac = med_factors(j);
                    w2 = fac*med;
                    Ks{i,j} = KGGaussian(ewidth, w2);
                end
            end
            Kcell = reshape(Ks, [1, length(embed_widths)*length(med_factors)]);
        end

        function KCell = combineCandidatesAvgCov(kerConstructFunc, T, medf, subsamples)
            assert(isa(T, 'TensorInstances'));
            assert(isnumeric(medf));
            assert(~isempty(medf));
            assert(all(medf>0));
            if nargin < 3
                subsamples = 3000;
            end
            numInput=T.tensorDim();
            % Always embed with widths given by average covariance.
            % We will only vary the outer Gaussian widths
            embed_width2s=zeros(1, numInput);
            for i=1:numInput
                da=T.instancesCell{i};
                avgCov=RFGEProdMap.getAverageCovariance(da, subsamples);
                % keep just one number, the mean, instead of the diagonal covariance.
                embed_width2s(i)=mean(diag(avgCov));
            end

            % total number of candidats = len(medf)^numInput
            % Total combinations can be huge ! Be careful. Exponential in the 
            % number of inputs
            totalComb = length(medf)^numInput;
            KCell = cell(1, totalComb);
            % temporary vector containing indices
            I = cell(1, numInput);
            for ci=1:totalComb
                [I{:}] = ind2sub( length(medf)*ones(1, numInput), ci);
                II=cell2mat(I);
                inputWidth2s= medf(II).*embed_width2s;
                kers = cell(1, numInput);
                for ki=1:numInput
                    kers{ki} = KGGaussian(embed_width2s(ki), inputWidth2s(ki));
                end
                KCell{ci} = kerConstructFunc(kers);
            end
        end

        function KSumCell = ksumCandidatesAvgCov(T, medf, subsamples)
            kerConstructFunc = @(kers)KSum(kers);
            KSumCell = KGGaussian.combineCandidatesAvgCov(kerConstructFunc, ...
                T, medf, subsamples);
        end


        function KProductCell = productCandidatesAvgCov(T, medf, subsamples )
            % - Generate a cell array of KProduct candidates from medf,
            % a list of factors to be  multiplied with the 
            % diagonal of the average covariance matrices.
            %
            % - subsamples can be used to limit the samples used to compute
            % the average
            %
            kerConstructFunc = @(kers)KProduct(kers);
            KProductCell = KGGaussian.combineCandidatesAvgCov(kerConstructFunc, ...
                T, medf, subsamples);

        end  % end productCandidates

    end
end

