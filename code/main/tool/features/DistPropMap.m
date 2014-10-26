classdef DistPropMap < FeatureMap 
    %DISTPROPMAP Generate features from TensorInstances inputs by stacking 
    %many properties of the distributions e.g., entroby, variance, mode, etc.
    %   .
    
    properties(SetAccess=protected)
        % A cell array of distribution types of input variables 
        % e.g., {'DistBeta', 'DistNormal'}
        inputVarTypes;

        % #features for 1D Gaussian
        dnormal1dNumFeatures;
        dbetaNumFeatures;

        % total number of features
        totalFeatures;
    end
    
    methods
        function this=DistPropMap(inputVarTypes)
            this.inputVarTypes = inputVarTypes;
            assert(all(ismember(inputVarTypes, {'DistNormal', 'DistBeta'})));

            normalF = this.genFeaturesDistNormal1D(DistNormal(0, 1));
            this.dnormal1dNumFeatures = length(normalF);
            this.dbetaNumFeatures = length(this.genFeaturesDistBeta(DistBeta(1, 2)));

            total = 0;
            for i=1:length(inputVarTypes)
                if strcmp(inputVarTypes{i}, 'DistNormal')
                    total = total + this.dnormal1dNumFeatures;
                elseif strcmp(inputVarTypes{i}, 'DistBeta')
                    total = total + this.dbetaNumFeatures;
                else
                    error('Unknown distribution type: %s', inputVarTypes{i});
                end
            end
            this.totalFeatures = total;
            
        end

        function Z=genFeatures(this, T)
            % Z = numFeatures x n
            if isempty(T)
                Z = [];
                return;
            end
            
            assert(isa(T, 'TensorInstances') ); 
            d = T.tensorDim();
            assert(d == length(this.inputVarTypes));
            ic = T.instancesCell;
            features = cell(1, d);
            for i=1:length(ic)
                dai = ic{i};
                assert(isa(dai, 'DistArray'));
                daDim = unique(dai.d);
                if isa(dai(1), 'DistNormal') && daDim == 1
                    % 1d Gaussian 
                    F = this.genFeaturesDistNormal1D(dai);
                elseif isa(dai(1), 'DistBeta')
                    F = this.genFeaturesDistBeta(dai);
                else
                    error('Unknown distribution type %s', class(dai(1)));
                end
                features{i} = F;
            end
            Z = vertcat(features{:});

        end

        function M=genFeaturesDynamic(this, T)
            % Generate feature vectors in the form of DynamicMatrix.
            assert(isa(T, 'TensorInstances'));
            Z = this.genFeatures(T);
            M = DefaultDynamicMatrix.fromMatrix(Z);
        end

        function g=getGenerator(this, T)
            g=@(I, J)this.generator(T, I, J);

        end

        function Z=generator(this, T, I, J )
            % I=indices of features, J=sample indices 
            assert(isa(T, 'TensorInstances'));
            RT=T.instances(J);
            % generate more than needed if I is not all rows.
            % Not that expensive. It is okay ...
            Fea=this.genFeatures(RT);
            Z = Fea(I, :);
        end

        function fm=cloneParams(this, numFeatures)
            % numFeatures is fixed.
            fm = DistPropMap(this.inputVarTypes);
        end

        function D=getNumFeatures(this)
            D = this.totalFeatures;
        end

        function s=shortSummary(this)
            s = sprintf('%s', mfilename);
        end

        % from PrimitiveSerializable interface
        function s=toStruct(this)
            s.className=class(this);
            s.inputVarTypes = this.inputVarTypes;
        end

        function F=genFeaturesDistNormal1D(this, distArrayNormal)
            assert(unique(distArrayNormal.d)==1);
            assert(isa(distArrayNormal(1), 'DistNormal'));
            da = DistArray(distArrayNormal);
            M = [da.mean];
            V = [da.variance];
            M2 = V + M.^2;
            % natural parameters
            %Nat1 = M./V;
            Nat2 = -0.5./V;
            Ent = 0.5*log(2*pi*exp(1)*V);
            % quantile points 
            qPoints = [0.1, 0.25, 0.75, 0.9]';
            Quan = repmat(M, length(qPoints), 1) + erfinv(2*qPoints - 1)*sqrt(2*V);
            % stack all features
            Stat = [M; V; M2;  Nat2; Ent; Quan ];
            s = size(Stat, 1);
            Stat2Prod = MatUtils.colOutputProduct(Stat, Stat);
            Stat2Re = reshape(Stat2Prod, [s^2, size(Stat, 2)]);
            Stat2 = Stat2Re(find(tril(ones(s))), :);
            F = [Stat; Stat2 ];
            

        end

        function F=genFeaturesDistBeta(this, distArrayBeta)
            da = DistArray(distArrayBeta);
            % See http://en.wikipedia.org/wiki/Beta_distribution
            assert(isa(da(1), 'DistBeta'));
            M = [da.mean];
            V = [da.variance];
            M2 = V + M.^2;
            % natural parameters
            inDa = da.distArray;
            A = [inDa.alpha];
            B = [inDa.beta];
            %Nat1 = A-1;
            %Nat2 = B-1;
            %Mode = (A-1)./(A + B -2); % only for alpha, beta > 1
            AB1 = A+B+1;
            AB2 = A+B+2;
            Skew = 2*(B-A).*sqrt(AB1)./( AB2.*sqrt(A.*B));
            Kurtosis = 6*( ((A-B).^2).*AB1 - A.*B.*AB2  )./( A.*B.*AB2.*(A+B+3) );
            PsiA = psi(A);
            PsiB = psi(B);
            PsiAB = psi(A+B);
            Ent = betaln(A, B) - (A-1).*PsiA - (B-1).*PsiB + (A+B-2).*PsiAB;
            Elnx = PsiA - PsiAB;
            Vlnx = psi(1, A) - psi(1, A+B);

            % stack all features
            Stat = [M; V; M2; A; B; Skew; Kurtosis; Ent; Elnx; Vlnx];
            %s = size(Stat, 1);
            %Stat2Prod = MatUtils.colOutputProduct(Stat, Stat);
            %Stat2Re = reshape(Stat2Prod, [s^2, size(Stat, 2)]);
            %Stat2 = Stat2Re(find(tril(ones(s))), :);
            %F = [Stat; Stat2 ];
            F=Stat;

        end

    end
    
    methods(Static)
    end

end

