classdef DistNormalBuilder < DistBuilder
    %DISTNORMALBUILDER DistBuilder for DistNormal
    %    Work for 1d and multivariate Gaussians.
    
    properties
    end
    
    methods
      
        function L=empty(this, r, c)
            L = DistNormal.empty(r, c);
        end
        
        function S=getStat(this, D)
            assert(isa(D, 'DistNormal') || isa(D, 'DistArray'));
            assert(~isa(D, 'DistArray') || isa(D.distArray, 'DistNormal'));
            dim = unique(D.d);
            assert(length(dim)==1);

            M1 = [D.mean];
            if dim==1
                M2 = [D.variance] + M1.^2;
                S = [M1; M2];
            else
                % multivariate 
                M1outer = MatUtils.colOutputProduct(M1, M1);
                M2 = cat(3, D.variance) + M1outer;
                n = length(D);
                S = [M1; reshape(M2, [dim^2, n])];
            end

        end
        
        function D=fromStat(this, S)
            % Construct DistNormal(s) from a sufficient statistics returned
            % from getStat()
            l = size(S, 1);
            dim = DistNormalBuilder.findDim(l);
            if isempty(dim)
                error('Invalid dimension of stat.');
            end
            n = size(S, 2);
            assert(length(dim)==1);
            if dim==1
                M = S(1,:);
                V = S(2,:)-M.^2;
                D = DistNormal(M, V);
            else
                dim = DistNormalBuilder.findDim(size(S, 1));
                M = S(1:dim, :);
                M2 = reshape(S((dim+1):end, :), [dim, dim, n]);
                M1outer = MatUtils.colOutputProduct(M, M);
                V = M2 - M1outer;
                clear M1outer;
                D = DistNormal(M, V);
            end
        end

        function Mcell=getMoments(this, D)
            assert(isa(D, 'DistNormal'));
            n=length(D);
            Mcell=cell(1, n);
            for i=1:n
                m1=D(i).mean(:);
                m2=D(i).variance+ m1*m1';
                Mcell{i}={m1, m2};
            end
        end

        function D=fromMoments(this, Mcell)
            assert(iscell(Mcell));
            n=length(Mcell);
            D=DistNormal.empty(0, n);
            for i=1:n
                m=Mcell{i}{1};
                v=Mcell{i}{2}-m*m';
                D(i)=DistNormal(m, v);
            end
        end
        
        function D= fromSamples(this, samples, weights)
            assert(size(samples, 1)==1, 'Only 1d Gaussian is supported');
            S = this.suffStat(samples);
            SW = S*weights(:);
            m = SW(1);
            v = SW(2)-m^2;
            D = DistNormal(m ,v);
        end
        
        function s = shortSummary(this)
            s = mfilename;
        end

        % From PrimitiveSerializable interface 
        function s=toStruct(this)
            s.className=class(this);
        end
        %%%%%%%%%%%%%%%%%%%%5
        
        function S=suffStat(this, X)
            % phi(x)=[x, x^2]' or phi(x)=[x; vec(xx')]
            % X (dxn) contains samples.
            [d,n] = size(X);
            assert(d>=1)
            if d==1
                S = [X; X.^2];
            else
                % multivariate
                M2 = MatUtils.colOutputProduct(X, X);
                S = [X; reshape(M2, [d^2, n])];
            end
            
        end 
%         

%         function T=stableSuffStat(this, S)
%             assert(size(S,1)==2, 'Only univariate Gaussian is supported');
%             M = S(1,:);
%             V = S(2,:)-M.^2;
%             Mok = isfinite(M);
%             Vok = isfinite(V) & abs(V)>1e-4 ;
%             T = Mok & Vok;
%         end
%         
%         function obj=dummyObj(this)
%             obj = DistNormal(nan, nan);
%         end
    end
    
    methods (Static)
        function d=findDim(l)
            % Find d given l = d^2 + d where d,l are integers
            % 
            sq = 1:ceil(sqrt(l));
            D = sq.*(sq+1);
            i = find(l==D);
            if isempty(i)  
                d=[];
            else
                d = sq(i);
            end

        end
    end
end

