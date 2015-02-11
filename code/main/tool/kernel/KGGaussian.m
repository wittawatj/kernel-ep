classdef KGGaussian < Kernel & PrimitiveSerializable
    %KGGAUSSIAN Kernel for distributions defined as the Gaussian kernel on
    %the mean embeddings into Gaussian RKHS.
    %
    % Outer Gaussian kernel with parameter w2. The distributions
    % in X1, X2 are also embedded with a Gaussian kernel with parameter
    % embed_width2s
    %
    % Input: X1, X2 = 1xn array of DistNormal.
    %
    properties (SetAccess=private)
        % Gaussian kernel for embedding
        kegauss;
        % embedding width2 for mean embeddings. 
        % one parameter for each dimension of the  input distributions. 
        embed_width2s;
        % width2 for the outer Gaussian kernel on the mean embeddings.
        width2;
        
    end
    
    methods
        function this=KGGaussian(embed_width2s, width2)
            % sigm2 = Width for embedding into Gaussian RKHS
            % width2 = Gaussian width^2. Not the one for embedding the
            % distribution.
            assert(isscalar(width2));
            assert(width2 > 0, 'Gaussian width must be > 0');
            assert(all(embed_width2s >0));
            % KEGaussian supports one width for each dimension 
            this.kegauss = KEGaussian(embed_width2s);
            this.embed_width2s = embed_width2s;
            this.width2 = width2;
        end
        
        
        function Kmat = eval(this, P, Q)
            assert(isa(P, 'Distribution'));
            assert(isa(Q, 'Distribution'));
            dim_p = unique([P.d]);
            assert(length(dim_p) == 1, 'Dimensionally inhomogenous dist array P.');
            dim_q = unique([Q.d]);
            assert(length(dim_q) == 1, 'Dimensionally inhomogenous dist array Q.');
            assert(dim_p == length(this.embed_width2s), 'param length does not match dimension of P');
            assert(dim_q == length(this.embed_width2s), 'param length does not match dimension of P');

            pp = this.kegauss.pairEval(P, P);
            qq = this.kegauss.pairEval(Q, Q);
            pq = this.kegauss.eval(P, Q);
            D2 = bsxfun(@plus, pp(:), qq(:)') - 2*pq;
            Kmat = exp( -D2/(2*this.width2) );
        end
        
        
        function Kvec = pairEval(this, X, Y)
            % If X, Y are not Gaussian, we will treat them as one by doing moment 
            % matching i.e., extract mean and variance and construct a Gaussian 
            % out of them.
            assert(isa(X, 'Distribution'));
            assert(isa(Y, 'Distribution'));
            assert(length(X)==length(Y));
            
            pp = this.kegauss.pairEval(X, Y);
            qq = this.kegauss.pairEval(X, Y);
            pq = this.kegauss.pairEval(X, Y);
            D2 = pp + qq - 2*pq;
            Kvec = exp(-D2/(2*this.width2));
            %if X(1).d==1
            %    % operation on obj array can be expensive..
            %    M1 = [X.mean];
            %    V1 = [X.variance];
            %    M2 = [Y.mean];
            %    V2 = [Y.variance];
                
            %    T1 = KEGauss1.self_inner1d(M1, V1, sigma2);
            %    T2 = KEGauss1.self_inner1d(M2, V2, sigma2);
            %    Cross = this.kegauss.pairEval(X, Y);
            %    Kvec = exp(-(T1-2*Cross+T2)/(2*w2) );
            %else
            %    error('later for multivariate');
            %end
        end
        
        function Param = getParam(this)
            Param = {this.embed_width2s, this.width2};
        end
        
        function s=shortSummary(this)
            s = sprintf('%s(%.2g, %.2g)', mfilename, this.embed_width2s, this.width2 );
        end

        % from PrimitiveSerializable interface
        function s=toStruct(this)
            %kegauss;
            %embed_width2s;
            %width2;
            s = struct();
            s.className=class(this);
            s.kegauss = this.kegauss.toStruct();
            s.embed_width2s = this.embed_width2s;
            s.width2 = this.width2;
        end
    end
    
    methods (Static)
        
        function [ D2] = distGGaussian( X1, X2, embed_width2s)
            %DISTGGAUSSIAN Distance^2 matrix before taking the exponential.
            % The actual formula is for Gaussian distributions. 
            % If X, Y are not Gaussian, treat them as one by moment matching.
            %
            %[~, n1] = size(X1);
            %[~, n2] = size(X2);
            assert(isa(X1, 'Distribution'));
            assert(isa(X2, 'Distribution'));
            kegauss = KEGaussian(embed_width2s);

            pp = kegauss.pairEval(X1, X1);
            qq = kegauss.pairEval(X2, X2);
            pq = kegauss.eval(X1, X2);
            D2 = bsxfun(@plus, pp(:), qq(:)') - 2*pq;
        end

        function [DD, M]=compute_meddistances(X, xembed_widths, subsamples)
            % for every embed width, compute the pairwise median distance
            % xembed_widths in a list.
            % High storage cost for DD.
            % - subsamples is an integer denoting the size of subsamples
            % that will be used instead to compute the meddistance.
            %
            % Output:
            %  - DD{i}: a pairwise distance with embedding width2 xembed_widths{i} 
            %  - M(i): median of DD{i}
            %
            assert(isa(X, 'Distribution'));
            assert(iscell(xembed_widths));

            X = DistArray(X);
            if nargin >=3 && subsamples < length(X)
                I = randperm(length(X), subsamples);
                X = X.get(I); %not DistArray anymore
            end

            M = zeros(1, length(xembed_widths));
            DD = cell(1, length(xembed_widths));
            for i=1:length(xembed_widths)
                embed2 = xembed_widths{i};
                dim = unique([X.d]);
                assert(dim == length(embed2), 'Embedding width vector does not match dim. of input');
                D2 = KGGaussian.distGGaussian( X, X, embed2);
                DD{i} = D2;
                M(i) = median(D2(:));
            end

        end

        function Kcell = candidates(X, embed_width2s_cell, med_factors, subsamples)
            % Generate a cell array of KGGaussian candidates from a list of
            % embeding widths. 
            % - embed_width2s_cell is a cell array where each element is a vector 
            % embed_width2s 
            % - med_factors: list of factors
            % to be  multipled with the pairwise median distance of the mean
            % embeddings.
            %
            % - subsamples can be used to limit the samples used to compute
            % median distance.

            if nargin < 4
                subsamples = min(2000, length(X));
            end
            assert(isa(X, 'Distribution'));
            assert(iscell(embed_width2s_cell));
            assert(~isempty(embed_width2s_cell));
            assert(isnumeric(med_factors));
            assert(~isempty(med_factors));
            assert(all(med_factors > 0));

            Ks = cell(length(embed_width2s_cell), length(med_factors));
            for i=1:length(embed_width2s_cell)
                ewidth = embed_width2s_cell{i};
                [~, med]= KGGaussian.compute_meddistances(X, {ewidth}, subsamples);
                for j=1:length(med_factors)
                    fac = med_factors(j);
                    w2 = fac*med;
                    Ks{i,j} = KGGaussian(ewidth, w2);
                end
            end
            Kcell = reshape(Ks, [1, length(embed_width2s_cell)*length(med_factors)]);
        end

        function KCell = combineCandidatesAvgCov(kerConstructFunc, T, medf, subsamples)
            assert(isa(T, 'TensorInstances'));
            assert(isnumeric(medf));
            assert(~isempty(medf));
            assert(all(medf>0));
            if nargin < 4
                subsamples = 3000;
            end
            numInput=T.tensorDim();
            % Always embed with widths given by average covariance.
            % We will only vary the outer Gaussian widths
            embed_width2s_cell=cell(1, numInput);
            for i=1:numInput
                da=T.instancesCell{i};
                avgCov=RFGEProdMap.getAverageCovariance(da, subsamples);
                % keep just one number, the mean, instead of the diagonal covariance.
                embed_width2s_cell{i} = diag(avgCov);
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
                % TODO: This is a strange heuristic ...
                inputWidth2s_cell = cellfun(@(width2s, med) (width2s*med), embed_width2s_cell, num2cell(medf(II)));
                %inputWidth2s= medf(II).*embed_width2s;
                kers = cell(1, numInput);
                for ki=1:numInput
                    kers{ki} = KGGaussian(embed_width2s_cell{ki}, inputWidth2s_cell{ki});
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

