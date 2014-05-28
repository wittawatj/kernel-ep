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
            
%             fplot( @(x)pdf('gamma', x,  1 ,60), [0 , 800])
            con = gamrnd(1, 30, 1, N);
            from = 0.01;
            peakLocs = rand(1, N)*(1-from*2) + from;
            AX = peakLocs.*con;
            BX = con-AX;
            X = DistBeta(AX, BX);
            assert(all([X.mean]>=from & [X.mean]<=1-from) )
            
            % gen T
            MT = randn(1, N)*sqrt(20);
            VT = unifrnd(0.01, 500, 1, N);
            T = DistNormal(MT, VT);
            
            % proposal distribution for for the conditional varibles (i.e. t)
            % in the factor. Require: Sampler & Density.
            op.in_proposal = DistNormal( 0, 30);
            
            % A forward sampling function taking samples (array) from in_proposal and
            % outputting samples from the conditional distribution represented by the
            % factor.
            op.cond_factor = @SigmoidFactor.sigmoid;
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
            op.iw_samples = 1e5;
            op.seed = seed;
            
            op.sample_cond_msg = true;
            op.left_distbuilder = [];
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
        
        function runTestKMVMapper(op)
            % Test the mean embedding operator defined with 2 KMVGauss1 kernels. 
            % One on Beta incoming message, the other on Gaussian incoming
            % message. KMVGauss1 relies on mean and variance of messages.
            %
            if nargin < 1
                op = [];
            end
            op.mapper_learner = myProcessOptions(op, 'mapper_learner', ...
                @DistMapper2Factory.learnKMVMapper1D );
            op.n_loaded_samples = myProcessOptions(op, 'n_loaded_samples', 4000);
            % important
            op.message_set_loader = myProcessOptions(op, 'message_set_loader', ...
                @SigmoidFactor.l_sigmoidTrainMsgs );
            
            % kernel parameters
            op.mean_med_factors = myProcessOptions(op, ...
                'mean_med_factors', [ 10]);
            op.variance_med_factors = myProcessOptions(op, ...
                'variance_med_factors', [ 10]);
            op.reglist = [1e-2, 1, 5];

            t_gauss1TensorMapper2In( op );
        end
        
        function [R] = runSigmoidEPWithKMV( op )
            % - Run sigmoid EP problem with KMVGauss1 kernel
            %
            if nargin < 1
                op=[];
            end
            error('EP part not complete yet. Need to fix distMapper2_ep to support Beta');
            
            op.seed = myProcessOptions(op, 'seed', 1);
            seed = op.seed;
            oldRng = rng();
            rng(seed);
            
            op.sigmoid_model_path = myProcessOptions(op, 'sigmoid_model_path', ...
                'saved/sigmoid_kmv_mapper.mat');
            
            % Boolean flag
            op.retrain_sigmoid_model = myProcessOptions(op, 'retrain_sigmoid_model', false);
            if ~exist(op.sigmoid_model_path, 'file') || op.retrain_sigmoid_model
                SigmoidFactor.trainSaveModel(op);
            else
                % mapper already exists in a file. load it.
                load(op.sigmoid_model_path);
                op = dealstruct(saved_op, op);
            end
            
            % new data set for testing EP. Not for learning an operator.
            op.observed_size = myProcessOptions(op, 'observed_size', 500);
            nN = op.observed_size;
            % the mean of theta
            op.clutter_theta_mean = myProcessOptions(op, 'clutter_theta_mean', 3);
            
            % [Theta, tdist] = Clutter.theta_dist(nN);
            Theta = randn(1, nN)*sqrt(0.1) + op.clutter_theta_mean ;
            [NX, xdist] = ClutterMinka.x_cond_dist(Theta, a, w);
            
            % start EP
            [ R] = distMapper2_ep( NX, mapper, op);
            
            % records
            op.data = s;
            R.op = op;
            % % Minka's EP
            % C = ClutterMinka(a, w);
            % % initial values for q
            % m0 = 0;
            % v0 = 10;
            % % TM (iterations x N) = mean of each i for each iteration t
            % [R] = C.ep(NX, m0, v0, seed);
            %
            % % Clutter problem solved with importance sampling projection
            % theta_proposal = DistNormal(2, 20);
            % iw_samples = 2e4;
            % IC = ClutterImportance(a, w, theta_proposal, iw_samples);
            % IR =  IC.ep( NX, m0, v0, seed);
            
            %%% plot
            % plot_clutter_results(tdist, xdist, tp_pdf, q, seed);
            % kernel EP
            % [h1,h2]= plot_epfacs( KR );
            % [h1,h2]= plot_epfacs( R );
            
            % keyboard
            %%%%%%%%%%%%%%%%%%%%%%%%%%5
            rng(oldRng);
        end
        
        
        
        
    end
    
end

