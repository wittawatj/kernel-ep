classdef SigmoidBundleGenerator < BundleGenerator & HasOptions
    %SIGMOIDBUNDLEGENERATOR Generate MsgBundle for sigmoid factor problem.
    
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
            kv.var2_distbuilder='DistBuilder for t in p(x|t)';
            kv.iw_samples='Number of importance weights to draw';
            kv.sample_cond_msg=['If true, sample from the conditioning variable '...
                'messages instead. If false, use in_proposal.'];
            kv.is_beta_observed=['If true, assume the Beta variable is observed. ',...
                'Only Beta(1,2) and Beta(2,1) are incoming messages from Beta var.']

            od=OptionsDescription(kv);
        end

        function Op=getDefaultOptions(this)
            st=struct();
            st.seed=1;
            %st.in_proposal=DistNormal(0, 15);
            st.in_proposal=Grid1DSampler(-20, 20);
            % X is beta in p(x|t)
            st.var1_distbuilder = DistBetaBuilder();
            % T is Gaussian in p(x|t)
            st.var2_distbuilder= DistNormalBuilder();
            st.iw_samples=2e4;
            % Should we sample_cond_msg ? 
            st.sample_cond_msg=false;
            % If true, assume Beta variable is observed. 
            st.is_beta_observed = false;

            Op=Options(st);
        end

        % generate a MsgBundle which can be used to train an operator
        function bundle=genBundle(this, n, varOutIndex)
            % index=1 -> x in p(x|t)
            assert(varOutIndex==1 || varOutIndex==2);
            op=this.getDefaultOptions().toStruct();
            is_beta_observed = this.opt('is_beta_observed');
            if varOutIndex==1
                op.right_distbuilder=[];
                [ X, T, Xout, Tout ]=SigmoidBundleGenerator.genOutBundles(n, op, is_beta_observed);
                bundle=DefaultMsgBundle(Xout, X, T);
                assert(isempty(Tout));

            else
                op.left_distbuilder=[];
                [ X, T, Xout, Tout ]=SigmoidBundleGenerator.genOutBundles(n, op, is_beta_observed);
                bundle=DefaultMsgBundle(Tout, X, T);
                assert(isempty(Xout));

            end

        end

        function fbundles=genBundles(this, n)

            op=this.options.toStruct();
            is_beta_observed = this.opt('is_beta_observed');
            [ X, T, Xout, Tout ]=SigmoidBundleGenerator.genOutBundles(n, op, is_beta_observed);
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

        function s=saveobj(this)
            s.options=this.options;
        end

    end
    
    methods(Static)
        function obj=loadobj(s)
            obj=SigmoidBundleGenerator();
            obj.options=s.options;
        end

        function s=sigmoid(z)
            s = 1./(1+exp(-z));
        end

        function [ X, T, Xout, Tout ]=genOutBundles(N, op, is_beta_observed)
            %Generate training set (messages) for sigmoid factor.
            %
            % T = theta
            % Tout = outgoing messages for theta (after projection)
            % Assume p(x|t) or x=p(t). x is a Beta. t is a Gaussian.
            %
            oldRng = rng();
            rng(op.seed);
            
            % gen X
            
%             fplot( @(x)pdf('gamma', x,  1 ,10), [0 , 100])
            %con = gamrnd(1, 10, 1, N);
            %from = 0.01;
            %peakLocs = rand(1, N)*(1-from*2) + from;
            %AX = peakLocs.*con;
            %BX = con-AX;
            %X = DistBeta(AX, BX);
            
            %AX = gamrnd(1, 20, 1, N);
            %BX = gamrnd(1, 20, 1, N);
            %
            if is_beta_observed
                allPoss = cell(1, 2);
                allPoss{1} = DistBeta(1, 2);
                allPoss{2} = DistBeta(2, 1);
                X = allPoss(randi([1, 2], 1, N));
                X = [X{:}];
            else
                AX = unifrnd(0.01, 20, 1, N);
                BX = unifrnd(0.01, 20, 1, N);
                X = DistBeta(AX, BX);
            end
            
            %X = DistBeta(repmat(2, 1, N), repmat(3, 1, N));
            %assert(all([X.mean]>=from & [X.mean]<=1-from) )
            
            % gen T
            %MT = randn(1, N)*sqrt(70); % cover -10 to 10
            MT = unifrnd(-17, 17, 1, N);
            VT = unifrnd(0.01, 200, 1, N);
            T = DistNormal(MT, VT);
            
            % A forward sampling function taking samples (array) from in_proposal and
            % outputting samples from the conditional distribution represented by the
            % factor.
            op.cond_factor = @SigmoidBundleGenerator.sigmoid;
            op.left_distbuilder=op.var1_distbuilder;
            op.right_distbuilder=op.var2_distbuilder;
            
            [ X, T, Xout, Tout ] = gentrain_dist2(X, T, op);

            assert(length(X)==length(T));
            assert(length(T)==length(Xout));
            assert(length(Xout)==length(Tout));

            rng(oldRng);
        end
    end % end static methods 
end

