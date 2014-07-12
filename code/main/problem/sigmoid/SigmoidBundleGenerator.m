classdef SigmoidBundleGenerator < BundleGenerator & HasOptions
    %SIGMOIDBUNDLEGENERATOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess=protected)
        % instance of Options
        options;
    end
    
    methods
        function this=SigmoidBundleGenerator()
            this.options=this.getDefaultOptions();
        end
        
        % Return an instance of OptionsDescription describing possible options.
        function od=getOptionsDescription(this)
            % key-value pairs of open-description
            kv=struct();
            kv.seed='random seed';
            kv.in_proposal=['proposal distribution for conditioning variable'...
                 'A class extending Sampler & Density'];
            kv.var1_distbuilder='DistBuilder for x in p(x|t)';
            kv.var2_distbuilder='DistBetaBuilder for t in p(x|t)';
            kv.iw_samples='Number of importance weights to draw';
            kv.sample_cond_msg=['If true, sample from the conditioning variable '...
                'messages instead. If false, use in_proposal.'];

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            st.in_proposal=DistNormal(0, 6);
            % X is beta in p(x|t)
            st.var1_distbuilder = DistBetaBuilder();
            % T is Gaussian in p(x|t)
            st.var2_distbuilder= DistNormalBuilder();
            st.iw_samples=2e5;
            % Should we sample_cond_msg ? 
            st.sample_cond_msg=true;

            Op=Options(st);
        end

        % generate a MsgBundle which can be used to train an operator
        function bundle=genBundle(this, n, varOutIndex)
            % index=1 -> x in p(x|t)
            assert(varOutIndex==1 || varOutIndex==2);
            op=this.getDefaultOptions().toStruct();
            if varOutIndex==1
                op.right_distbuilder=[];
                [ X, T, Xout, Tout ]=SigmoidBundleGenerator.genOutBundles(n, op);
                bundle=DefaultMsgBundle(Xout, {X, T});
                assert(isempty(Tout));

            else
                op.left_distbuilder=[];
                [ X, T, Xout, Tout ]=SigmoidBundleGenerator.genOutBundles(n, op);
                bundle=DefaultMsgBundle(Tout, {X, T});
                assert(isempty(Xout));

            end

        end

        function fbundles=genBundles(this, n)

            op=this.getDefaultOptions().toStruct();
            [ X, T, Xout, Tout ]=SigmoidBundleGenerator.genOutBundles(n, op);
            Xda=DistArray(X);
            Tda=DistArray(T);
            Xoutda=DistArray(Xout);
            Toutda=DistArray(Tout);
            fbundles=FactorBundles({Xda, Tda}, {Xoutda, Toutda});
        end

        function nv=numVars(this)
            nv=2;
        end

        %s is a string describing this generator.
        function s=shortSummary(this)
            s=mfilename;
        end

    end
    
    methods(Static)
        function s=sigmoid(z)
            s = 1./(1+exp(-z));
        end

        function [ X, T, Xout, Tout ]=genOutBundles(N, op)
            %Generate training set (messages) for sigmoid factor.
            %
            % T = theta
            % Tout = outgoing messages for theta (after projection)
            % Assume p(x|t) or x=p(t). x is a Beta. t is a Gaussian.
            %
            oldRng = rng();
            rng(op.seed);
            
            % gen X
            
            % If we observe an x=0.4, say, we want to represent it with a
            % Beta distribution with a peak at 0.4 and with low variance.
            % This can be done by setting alpha=x*1000, beta=1000-alpha.
            % The constant 1000 can be changed to something else. The
            % higher the lower the variance.
            %
%             con = 200;
            % Uniformly random peak locations in [from, 1-from]
            % Should focus on low variance because we will observe X and represent it
            % with a Beta which is close to a PointMass
            
%             fplot( @(x)pdf('gamma', x,  1 ,10), [0 , 100])
            con = gamrnd(1, 10, 1, N);
            from = 0.01;
            peakLocs = rand(1, N)*(1-from*2) + from;
            AX = peakLocs.*con;
            BX = con-AX;
            X = DistBeta(AX, BX);
            assert(all([X.mean]>=from & [X.mean]<=1-from) )
            
            % gen T
            MT = randn(1, N)*sqrt(4);
            VT = unifrnd(0.01, 100, 1, N);
            T = DistNormal(MT, VT);
            
            % A forward sampling function taking samples (array) from in_proposal and
            % outputting samples from the conditional distribution represented by the
            % factor.
            op.cond_factor = @SigmoidBundleGenerator.sigmoid;
            op.left_distbuilder=op.var1_distbuilder;
            op.right_distbuilder=op.var2_distbuilder;
            
            [ X, T, Xout, Tout ] = gentrain_dist2(X, T, op);
            rng(oldRng);
        end
    end
end

