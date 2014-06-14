classdef SigmoidFactor
    %SIGMOIDFACTOR A deterministic sigmoid factor.
    %   Assume convention p(x|t) where p(.|t) is a delta distribution
    %   because the factor is deterministic.
    
    properties
    end
    
    methods
    end
    
    properties (Constant)
        DEFAULT_DATA_PATH = 'saved/sigmoidTrainMsgs.mat';
        DEFAULT_BUNDLE_PATH = 'saved/sigmoidMsgBundle.mat';
    end
    
    methods (Static)
        
        function s=sigmoid(z)
            s = 1./(1+exp(-z));
        end
        
        function [ X, T, Xout, Tout, op ] = genData(op)
            %Generate training set (messages) for sigmoid factor.
            %
            % T = theta
            % Tout = outgoing messages for theta (after projection)
            % Assume p(x|t) or x=p(t). x is a Beta. t is a Gaussian.
            %
            
            if nargin < 1
                op = [];
            end
            oldRng = rng();
            op.seed = myProcessOptions(op, 'seed', 1);
            rng(op.seed);
            
            N = myProcessOptions(op, 'generate_size', 1000);
            op.generate_size = N;
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
            MT = randn(1, N)*sqrt(3.5);
            VT = unifrnd(0.01, 10, 1, N);
            T = DistNormal(MT, VT);
            
            % proposal distribution for for the conditional varibles (i.e. t)
            % in the factor. Require: Sampler & Density.
            op.in_proposal = DistNormal( 0, 4);
            
            % A forward sampling function taking samples (array) from in_proposal and
            % outputting samples from the conditional distribution represented by the
            % factor.
            op.cond_factor = @SigmoidFactor.sigmoid;
            
            % X is beta in p(x|t)
            op.left_distbuilder = DistBetaBuilder();
            % T is Gaussian in p(x|t)
            op.right_distbuilder = DistNormalBuilder();
            [ X, T, Xout, Tout ] = gentrain_dist2(X, T, op);
            
            rng(oldRng);
        end
        
        function  g_sigmoidTrainMsgs(n , op )
            %Generate training messages for sigmoid factor and save
            %
            if nargin < 2
                op=[];
            end
            
            seed = myProcessOptions(op, 'seed', 1);
            oldRng = rng();
            rng(seed);
            
            sigmoid_data_path = myProcessOptions(op, 'sigmoid_data_path', ...
                SigmoidFactor.DEFAULT_DATA_PATH);
            
            % generate some data
            % options
            op.generate_size = n;
            op.iw_samples = 12e4;
            op.seed = seed;
            
            op.sample_cond_msg = true;
     
            % generate training set
            [ X, T, Xout, Tout, op ] = SigmoidFactor.genData(op);
            
            % sort dataset by the means of Tout
            [Tout_means, I] = sort([Tout.mean]);
            X = X(I);
            T = T(I);
            %             Xout = Xout(I);
            Tout = Tout(I);
            
            save(sigmoid_data_path, 'n', 'op',  'X', 'T',  'Tout');
            rng(oldRng);
        end
        
        function g_sigmoidMsgBundle(n ,op)
            % generate message bundle MsgBundle.
            if nargin < 2
                op = [];
            end
               
            seed = myProcessOptions(op, 'seed', 1);
            oldRng = rng();
            rng(seed);
            
            sigmoid_bundle_path = myProcessOptions(op, 'sigmoid_bundle_path', ...
                SigmoidFactor.DEFAULT_BUNDLE_PATH);
            
            % generate some data
            % options
            op.generate_size = n;
            op.iw_samples = 12e4;
            op.seed = seed;
            
            op.sample_cond_msg = true;
     
            % generate training set. p(x|t)
            [ X, T, Xout, Tout, op ] = SigmoidFactor.genData(op);
            % Assemble into a MsgBundle
            msgBundle2Right = DefaultMsgBundle2(X, T, Tout);
            msgBundle2Left = DefaultMsgBundle2(X, T, Xout);
            generateMethod = func2str(@SigmoidFactor.g_sigmoidMsgBundle);
            generateTime = clock();
            
            save(sigmoid_bundle_path,  'op',  'msgBundle2Left', ...
                'msgBundle2Right', 'generateMethod', 'generateTime');
            rng(oldRng);
            
        end
        
        function [ s] = l_sigmoidTrainMsgs( nload, op )
            %Load training messages for sigmoid factor
            % nload = number of instances to load
            %
            if nargin < 2
                op = [];
            end
            
            seed = myProcessOptions(op, 'seed', 2);
            
            % path to load data
            sigmoid_data_path = myProcessOptions(op, 'sigmoid_data_path', ...
                SigmoidFactor.DEFAULT_DATA_PATH);
            
            oldRng = rng;
            rng(seed);
            
            assert(exist(sigmoid_data_path, 'file')~=0, 'File not exists: %s', sigmoid_data_path );
            load(sigmoid_data_path);
            % load() will load 'n', 'op', 'a', 'w', 'X', 'T', 'Xout', 'Tout'
            
            % In = tensor of X and T
            % XIns = ArrayInstances(X);
            XIns = MV1Instances(X);
            % TIns = ArrayInstances(T);
            TIns = MV1Instances(T);
            In = TensorInstances({XIns, TIns});
            
            % use remove. So we can keep the sorted inputs.
            toremove = length(X)-min(nload, length(X));
            Id = randperm( length(X),  toremove);
            X(Id) = [];
            T(Id) = [];
            if exist('Xout', 'var') && ~isempty(Xout)
                Xout(Id) = [];
            else
                Xout = [];
            end
            
            if exist('Tout', 'var') && ~isempty(Tout)
                Tout(Id) = [];
            else
                Tout = [];
            end
            
            % pack everything into a struct s
            s.op = op;
            s.X = X;
            s.T = T;
            s.Xout = Xout;
            s.Tout = Tout;
            
            s.XIns = XIns;
            s.TIns = TIns;
            s.In = In;
            
            rng(oldRng);
        end
        
        function trainSaveModel(op)
            % Train and save model
            % total samples to use
            op.total_samples = myProcessOptions(op, 'total_samples', 8000);
            n = op.total_samples;
            op.train_split = myProcessOptions(op, 'train_split', floor(0.8*n));
            ntr = op.train_split;
            nte = min(100, n-ntr);
            
            op.sigmoid_data_path = myProcessOptions(op, 'sigmoid_data_path', ...
                SigmoidFactor.DEFAULT_DATA_PATH);
            
            % Load training messages
            [ s] = SigmoidFactor.l_sigmoidTrainMsgs( n, op );
            
            % load 'n', 'op', 'a', 'w', 'X', 'T', 'Xout', 'Tout'
            %     eval(structvars(100, s));
            % load variables in s
            
            [X, T, Tout, Xte, Tte, Toutte] = Data.splitTrainTest(s, ntr, nte);
            assert(length(X)==ntr);
            assert(length(T)==ntr);
            assert(length(Tout)==ntr);
            
            % options for repeated hold-outs
            op.num_ho = 4;
            op.train_size = floor(0.7*ntr);
            op.test_size = min(2000, ntr - op.train_size);
            op.chol_tol = 1e-5;
            op.chol_maxrank = min(700, ntr);
            op.reglist = [1e-2, 1];
            % op.use_multicore = false;
            
            % options used in learnMapper
            op.med_subsamples = min(1500, ntr);
            op.mean_med_factors = [1];
            op.variance_med_factors = [1];
            
            % options for EP (for distMapper2_ep() )
            op.f0 = DistNormal(0, 30);
            op.ep_iters = 10;
            op.observed_variance = 0.1;
            op.mean_conv_thresh = 0.05;
            op.var_conv_thresh = 0.5;
            
            % learn a mapper from X to theta
            [mapper, C] = DistMapper2Factory.learnKMVMapper1D(X, T, Tout, op);
            saved_op = op;
            save(op.clutter_model_path, 'mapper', 'C', 'saved_op',  's');
        end
        
    
        
    end
    
end

