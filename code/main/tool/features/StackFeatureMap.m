classdef StackFeatureMap < FeatureMap 
    %STACKFEATUREMAP A meta FeatureMap taking a list of FeatureMap's and concatenating
    % all features.
    %   . s
    
    properties(SetAccess=protected)
        % a cell array of FeatureMap's
        featureMaps; 

    end
    
    methods
        function this=StackFeatureMap(fmapCell)
            % * fmapCell is a cell array of FeatureMap
            %
            %
            isMap = @(m)(isa(m, 'FeatureMap'));
            % ensure that all items are FeatureMap
            assert(all(cellfun(isMap, fmapCell)));
            this.featureMaps = fmapCell;

        end

        function Z=genFeatures(this, T)
            % Z = total numFeatures x n
            assert(isa(T, 'TensorInstances') ); 
            m = length(this.featureMaps);
            Zs = cell(1, m);
            for i=1:m
                Zs{i} = this.featureMaps{i}.genFeatures(T);
            end
            Z = vertcat(Zs{:});

        end

        function M=genFeaturesDynamic(this, T)
            % Generate feature vectors in the form of DynamicMatrix.
            assert(isa(T, 'TensorInstances'));
            g=@(I, J)this.generator(T, I, J);
            n=length(T);
            nf = this.getNumFeatures();
            M=DefaultDynamicMatrix(g, nf, n);
        end

        function g=getGenerator(this, T)
            g=@(I, J)this.generator(T, I, J);

        end

        function Z=generator(this, T, I, J )
            % I=indices of features, J=sample indices 
            assert(isa(T, 'TensorInstances'));
            lens = cellfun(@(fm)fm.getNumFeatures(), this.featureMaps);
            fmI = this.featureMapIndex(I);
            ends = cumsum(lens);
            ends2=[0, ends];
            localI = I - ends2(fmI);

            Z = nan(length(I), length(J));
            % fmI should be sorted at this point e.g., 1, 1, 2, 2, 3, 4, 4, 4..
            for u=unique(fmI(:)')
                ufmI = u==fmI;
                ulocal = localI(ufmI);
                g = this.featureMaps{u}.getGenerator(T);
                Z(ufmI, :) = g(ulocal, J);
            end
            assert(all(all(~isnan(Z))) );
        end

        function fm=cloneParams(this, numFeatures)
            m = length(this.featureMaps);
            fms = cell(1, m);
            for i=1:m
                fmi = this.featureMaps{i}.cloneParams(numFeatures);
                fms{i} = fmi;
            end
            fm = StackFeatureMap(fms);
        end

        function D=getNumFeatures(this)
            s = 0;
            m = length(this.featureMaps);
            for i=1:m
                s = s + this.featureMaps{i}.getNumFeatures();
            end
            D = s;
        end

        function s=shortSummary(this)
            s = sprintf('%s', mfilename);
        end
        
        % from PrimitiveSerializable interface
        function s=toStruct(this)
            s.className=class(this);
            d = length(this.featureMaps);
            structCell = cell(1, d);
            for i=1:d
                structCell{i} = this.featureMaps{i}.toStruct();
            end
            s.featureMaps = structCell;

        end
    end

    methods (Access=private)
        function fmI = featureMapIndex(this, I)
            % Convert the absolute indicies I into corresponding FeatureMap
            % indices.
            % fmI is a column vector
            
            %  FeatureMap lengths
            lens = cellfun(@(fm)fm.getNumFeatures(), this.featureMaps);
            ends = cumsum(lens);
            B = bsxfun(@le, I', ends);
            fmI = length(ends) - sum(B, 2)+1;
        end
    
    end
    
end

