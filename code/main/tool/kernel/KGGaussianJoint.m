classdef KGGaussianJoint < Kernel & PrimitiveSerializable
    %KGGAUSSIANJOINT Kernel for TensorInstances defined as the Gaussian kernel on
    %the mean embedding (with a Gaussian kernel) of the joint distribution formed 
    %by multiplying all incoming messages.
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
        % a KGGaussian kernel
        kggaussian;
    end
    
    methods
        function this=KGGaussianJoint(embed_width2, width2)
            % sigm2 = Width for embedding into Gaussian RKHS
            % width2 = Gaussian width^2. Not the one for embedding the
            % distribution.
            assert(width2 > 0, 'Gaussian width must be > 0');
            assert(embed_width2 >0);
            this.kggaussian = KGGaussian(embed_width2, width2);
        end
        
        
        function Kmat = eval(this, c1, c2)
            % Expect a cell of DistArray
            assert(iscell(c1));
            assert(iscell(c2));
            joint1Da = KGGaussianJoint.toJointDistArray(c1);
            joint2Da = KGGaussianJoint.toJointDistArray(c2);
            %             [Kmat, D2] = kerGGaussian(data1, data2, this.sigma2, this.width2);
            Kmat = this.kggaussian.eval(joint1Da, joint2Da);
        end
        
        function Kvec = pairEval(this, c1, c2)
            % If distributions in X, Yare not Gaussian, we will treat them as one by doing moment 
            % matching i.e., extract mean and variance and construct a Gaussian 
            % out of them.
            % - X, Y must be a cell of Distribution or DistArray

            joint1Da = KGGaussianJoint.toJointDistArray(c1);
            joint2Da = KGGaussianJoint.toJointDistArray(c2);
            Kvec = this.kggaussian.pairEval(joint1Da, joint2Da);
        end
        
        function Param = getParam(this)
            Param = this.kggaussian.getParam();
        end
        
        function s=shortSummary(this)
            s = this.kggaussian.shortSummary();
        end

        % from PrimitiveSerializable interface
        function s=toStruct(this)
            %kegauss;
            %embed_width2;
            %width2;
            s = struct();
            s.className=class(this);
            s.kggaussian = this.kggaussian.toStruct();
        end
    end
    
    methods (Static)
        function joint1Da = toJointDistArray(cellArray)
            % Convert a cell array of Distribution's to a DistArray of joint distributions
            %
            assert(iscell(cellArray));
            toDaFunc = @(D)(DistArray(D));
            c1 = cellfun(toDaFunc, cellArray, 'UniformOutput', false);

            T1 = TensorInstances(c1);
            % array of DistNormal
            joint1 = RFGJointEProdMap.tensorToJointGaussians(T1);
            joint1Da = DistArray(joint1);
        end
        
        function KCell = candidatesAvgCov( T, medf, subsamples)
            % This method is related to KGGaussian.combineCandidatesAvgCov
            assert(isa(T, 'TensorInstances'));
            assert(isnumeric(medf));
            assert(~isempty(medf));
            assert(all(medf>0));
            if nargin < 4
                subsamples = 4000;
            end
            jointGauss = KGGaussianJoint.toJointDistArray(T.instancesCell);
            % Dimension of the joint Gaussians.
            dim = unique([jointGauss.d]);
            assert(length(dim)==1);
            % Average covariance as a matrix.
            avgCov=RFGEProdMap.getAverageCovariance(jointGauss, subsamples);
            % Gaussian width-squared 
            % The average covariance will be multipled with median factor at the 
            % end.
            % TODO: We should consider one width for each dimension.
            %
            embed_width2s = mean(diag(avgCov));
            KCell = KGGaussian.candidates(jointGauss, embed_width2s, medf, subsamples);
            
        end % end candidatesAvgCov

    end
end

