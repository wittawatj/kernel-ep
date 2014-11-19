classdef MatchingPursuit < HasOptions
    %MATCHINGPURSUIT Perform matching pursuit given a dictionary of functions.
    %   - Assume that the output Y is in Euclidean space. 
    %   - Input X can be anything. Represented with an Instances object.
    %
    
    properties(SetAccess = protected)
        % An instance of Options
        options;

        % input samples. Instances.
        inputInstances;

        % output matrix. Each column is one output.
        outputMat;

        % W matrix of size dim(output) x b where b is the number of chosen 
        % basis functions.
        %weightMat = [];
    end

    properties(SetAccess=protected)
        %%%% internal properties %%%
        
        % dim(output) x n residual matrix.
        residualMat;
    end
    
    methods
        function this=MatchingPursuit(X, Y)
            this.options = this.getDefaultOptions();
            assert(isa(X, 'Instances'));
            assert(isnumeric(Y));
            assert(length(X) == size(Y, 2), 'Lengths of X and Y do not match');
            this.inputInstances = X;
            this.outputMat = Y;
        end

        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';
            kv.mp_function_classes = ['A cell array of MPFunctionClass s to consider'];
            kv.mp_reg = ['Regularization parameter for matching pursuit'];
            kv.mp_max_iters = ['Maximum iterations to perform matching pursuit'];
            kv.mp_backfit_every = ['Perform backfit for every specified iterations'];
            kv.mp_fc_subset = ['The max number of subsets of function classes ', ...
                'to consider in each iteration. Random subsampling function '...
                'class candidates.'];

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.mp_function_classes = [];
            st.mp_reg = 1e-5;
            st.mp_max_iters = 500;
            st.mp_backfit_every = 10;
            st.mp_fc_subset = 500;

            Op=Options(st);
        end

        function matchingPursuitIterate(this)
            % perform matching pursuit for one iteration from the current 
            % state. 
            %
            error('not implemented');
            if this.isNoKeyOrEmpty('mp_function_classes')
                error('mp_function_classes must be specified');
            end
        end

        function matchingPursuit(this)
            % perform full matching pursuit
            %
            X = this.inputInstances;
            n = length(X);
            Y = this.outputMat;
            % find best basis function g from each function class 

            if this.isNoKeyOrEmpty('mp_function_classes')
                error('mp_function_classes must be specified');
            end
            % regularization parameter
            mp_reg = this.opt('mp_reg');
            mp_function_classes = this.opt('mp_function_classes');
            mp_max_iters = this.opt('mp_max_iters');
            mp_backfit_every = this.opt('mp_backfit_every');

            nfc = length(mp_function_classes);
            fc_subset = min(nfc, this.opt('mp_fc_subset'));
            % initialization
            R = Y;
            for t=1:mp_max_iters
                if mod(t, mp_backfit_every) == 0
                    [W, GG] = this.backFit();
                    %display(sprintf('Backfit performed'));
                    assert(n == size(GG, 2));
                    % recompute R (residues)
                    R = Y - W*GG;
                end
                %
                % ### This can be parallelized.
                % best correlations to the residues for each function class
                bestCR = zeros(1, fc_subset);
                Gs = cell(1, fc_subset);
                mementos = cell(1, fc_subset);
                Wts = cell(1, fc_subset);
                Ifc = randperm(nfc, fc_subset);
                for i=1:fc_subset
                    fc = mp_function_classes{Ifc(i)};
                    [cr, G, wt, memento] = fc.findBestBasisFunction(R, mp_reg);
                    bestCR(i) = cr;
                    Gs{i} = G;
                    mementos{i} = memento;
                    Wts{i} = wt;
                end

                if all(cellfun(@isempty, Gs))
                    % no more candidates. Stop matching pursuit.
                    break;
                end
                % Find the best g  (basis function)
                [bestcr, gind] = max(bestCR);
                mp_function_classes{Ifc(gind)}.markBestBasisFunction(mementos{gind});
                G = Gs{gind};
                wt = Wts{gind};
                assert(length(G)==n);
                % total residua norms should reduce after every iteration.
                R = R - wt*G;
                display(sprintf('It: %d, residue: %.3f', t, norm(R, 'fro') ));
            end
            [W ] = this.backFit();

        end % end MatchingPursuit

        %function getCurrentIteration(this)
        %    % return current iteration number. At the beginning, return 0.
        %end

        function Fmat = evalFunction(this, X)
            % Evaluate the function using the selected bases on the specified 
            % samples X.
            % Return: F of size dim(output) x ntest
            %
            assert(isa(X, 'Instances'));
            mp_function_classes = this.opt('mp_function_classes');
            nfc = length(mp_function_classes);
            % ### This can be parallelized.
            dimy = size(this.outputMat, 1);
            ntest = length(X);
            Fmat = zeros(dimy, ntest);
            for i=1:nfc
                fc = mp_function_classes{i};
                % G = b x ntest
                Func = fc.evalFunction(X);
                Fmat = Fmat + Func;
            end
        end

        function [W, GG] = backFit(this)
            % Determine all W's by least squares
            %
            % Get matrix G i.e., evaluations of all selected basis functions on 
            % all samples. G is b x n.
            n = length(this.inputInstances);

            mp_function_classes = this.opt('mp_function_classes');
            nfc = length(mp_function_classes);
            Gs = cell(1, nfc);
            % ### This can be parallelized.
            for i=1:nfc
                fc = mp_function_classes{i};
                G = fc.evaluateOnTraining();
                Gs{i} = G;
            end
            % ### GG (b x n) can be huge !!
            GG = vertcat(Gs{:});
            b = size(GG, 1);
            X = this.inputInstances;
            Y = this.outputMat;
            assert(size(GG, 2) == length(X));
            % regularization parameter
            mp_reg = this.opt('mp_reg');
            %###  possible to use conjugate gradient here.
            opts = struct();
            opts.POSDEF = true;
            opts.SYM = true;

            sub = 1e5;
            subInd = randperm(n, min(n, sub));
            GGsub = GG(:, subInd);
            W = linsolve(GGsub*GGsub' + mp_reg*eye(b), GG*Y')';
            assert(size(W, 1) == size(Y, 1));

            % distribute subparts of W back to function classes
            si = 1;
            for i=1:nfc
                fc = mp_function_classes{i};
                basisCount = fc.countSelectedBases();
                subW = W(:, si:(si + basisCount-1) );
                fc.setWeightMatrix(subW);
                si = si + basisCount;
            end
            assert(si-1 == size(W, 2));
        end

    end %end methods
    
    methods (Static)

    end
end

