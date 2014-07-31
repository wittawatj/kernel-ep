classdef SimpleGaussBundleGenerator < BundleGenerator & HasOptions
    %SIMPLEGAUSSBUNDLEGENERATOR A generator for simple Gaussian problem.
    %    - The factor f(y|x) = N(y; x, I*var_y) 
    %    - The dataset is perhaps the simplest dataset to test a DistMapper.
    %    - Chang dimension of the Gaussian with opt('d')
    %
    
    properties (SetAccess=protected)
        % instance of Options
        options;
    end

    methods
        function this=SimpleGaussBundleGenerator()
            this.options=this.getDefaultOptions();
        end

        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';
            kv.var1_distbuilder='DistBuilder for y in p(y|x)';
            kv.var2_distbuilder='DistBuilder for x in p(y|x)';
            kv.iw_samples='Number of importance weights to draw';
            kv.sample_cond_msg=['If true, sample from the conditioning variable '...
                'messages instead. If false, use in_proposal.'];
            kv.d=['dimension of Gaussians'];
            kv.in_proposal=['proposal distribution for conditioning variable'...
                 'A class extending Sampler & Density'];
            kv.var_y=['scalar variance of y (isotropic Gaussian)'];

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            % DistBuilder for y in f(y|x)
            st.var1_distbuilder = DistNormalBuilder();
            st.var2_distbuilder= DistNormalBuilder();
            st.iw_samples=2e5;
            % Should we sample_cond_msg ? 
            st.sample_cond_msg=true;
            % Default to 1 dimension
            st.d=1;
            st.in_proposal=[];
            st.var_y=2;

            Op=Options(st);
        end

        % generate a MsgBundle which can be used to train an operator
        function bundle=genBundle(this, n, varOutIndex)
            % index=1 -> x in p(x|t)
            assert(varOutIndex==1 || varOutIndex==2);
            op=this.getDefaultOptions().toStruct();
            if varOutIndex==1
                op.right_distbuilder=[];
                [ Y, X, Yout, Xout]=SimpleGaussBundleGenerator.genOutBundles(n, op);
                bundle=DefaultMsgBundle(Yout, {Y, X});
                assert(isempty(Xout));

            else
                op.left_distbuilder=[];
                [Y, X, Yout, Xout]=SimpleGaussBundleGenerator.genOutBundles(n, op);
                bundle=DefaultMsgBundle(Xout, {Y, X});
                assert(isempty(Yout));

            end

        end

        function fbundles=genBundles(this, n)

            op=this.options.toStruct();
            [Y, X, Yout, Xout]=SimpleGaussBundleGenerator.genOutBundles(n, op);
            Yda=DistArray(Y);
            Xda=DistArray(X);
            Youtda=DistArray(Yout);
            Xoutda=DistArray(Xout);
            fbundles=FactorBundles({Yda, Xda}, {Youtda, Xoutda});
        end

        function nv=numVars(this)
            nv=2;
        end

        function s=shortSummary(this)
            %s is a string describing this generator.
            s=sprintf('%s(dim=%d)', mfilename, this.opt('d') );
        end

    end % end methods
    
    methods (Static)
        function [ Y, X, Yout, Xout]=genOutBundles(N, op)
            %Generate training set (messages) for simple Gaussian factor.
            %
            % - The factor f(y|x) = N(y; x, I*var_y)
            % - Y = Y messages (array of DistNormal)
            % - X= X messages (array of DistNormal)
            % - Yout = outgoing messages for Y (array of DistNormal)
            % - Xout = outgoing messages for X (array of DistNormal)
            %
            oldRng = rng();
            rng(op.seed);
            
            var_y=op.var_y;
            d=op.d;
            % gen X 
            % means scattered around 0
            MX=randn(d, N)*sqrt(10);
            VX=zeros(d, d, N);
            for i=1:N
                % varying degree of freedom
                df=randi([1, 10]);
                VX(:, :, i)=wishrnd(eye(d), df);
            end
            if d==1
                VX=squeeze(VX)';
            end
            X=DistNormal(MX, VX);

            % Gen Y. Same spec as  X
            MY=randn(d, N)*sqrt(10);
            VY=zeros(d, d, N);
            for i=1:N
                % varying degree of freedom
                df=randi([1, 10]);
                VY(:, :, i)=wishrnd(eye(d), df);
            end
            if d==1
                VY=squeeze(VY)';
            end
            Y=DistNormal(MY, VY);

            % A forward sampling function taking samples (array) from in_proposal and
            % outputting samples from the conditional distribution represented by the
            % factor.
            if isfield(op, 'in_proposal') || isempty(op.in_proposal)
                % degree of freedom of Wishart for variance of X is max at 10.
                % So we use 10 for variance of the proposal
                op.in_proposal=DistNormal(zeros(d, 1), eye(d)*10 );
            end

            function y=cond_factor(x)
                assert(size(x, 1)==d);
                % The factor does nothing but add some noise to x 
                nx=size(x, 2);
                y=randn(d, nx)*sqrt(var_y) + x;

            end
            op.cond_factor = @cond_factor;
            op.left_distbuilder=op.var1_distbuilder;
            op.right_distbuilder=op.var2_distbuilder;
            
            [ Y, X, Yout, Xout] = gentrain_dist2(Y, X, op);

            assert(length(X)==length(Y));
            assert(length(Y)==length(Xout));
            assert(length(Xout)==length(Yout));

            rng(oldRng);
        end
    end % end static methods

end

