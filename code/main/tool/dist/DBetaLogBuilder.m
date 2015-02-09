classdef DBetaLogBuilder < DistBetaBuilder
    %DBETALOGBUILDER DistBuilder for DistBeta. Use log(alpha) and log(beta).
    % - Actual beta incoming messages in the base of a logistic factor exhibit
    % a long tail distribution. Using log might be helpful.
    
    properties
    end
    
    methods
        function S=getStat(this, D)
            assert(isa(D, 'DistBeta') || isa(D, 'DistArray'));
            assert(~isa(D, 'DistArray') || isa(D.distArray, 'DistBeta'));
            if isa(D, 'DistArray')
                D = D.distArray;
            end
            la = log([D.alpha]);
            lb = log([D.beta]);
            S = [la; lb];
        end
        
        function D=fromStat(this, S)
            assert(size(S,1)==2, 'Expected two rows for log(alpha) and log(beta).');
            A = exp(S(1, :));
            B = exp(S(2, :));
            D = DistBeta(A, B);
        end

        function D= fromSamples(this, samples, weights)
            assert(size(samples, 1)==1, 'Beta samples must be 1d');
            assert(all(samples>=0 & samples <=1), 'Beta samples must be in [0,1].');
            
            % empirical mean
            m = samples*weights(:);
            % empirical 2nd moment
            m2 = (samples.^2)*weights(:);
            S = [m; m2];
            DistBetaBuilder builder = DistBetaBuilder();
            D = builder.fromStat( S );
        end
        
        function Scell = transformStat(this, X)
            error('later');
        end
        
        function s = shortSummary(this)
            s = mfilename;
        end
        
        % From PrimitiveSerializable interface 
        function s=toStruct(this)
            s.className=class(this);
        end
    end % end methods
    
end

