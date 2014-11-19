function [ ] = matching_pursuit_laplace( )
%MATCHING_PURSUIT_LAPLACE Test matching pursuit with Laplace function classes
%

seed=33;
oldRng=rng();
rng(seed, 'twister');

se=BundleSerializer();
bunName='sigmoid_bw_proposal_5000';
%bunName='sigmoid_bw_proposal_2000';
%bunName='sigmoid_bw_proposal_10000';
%bunName='sigmoid_bw_proposal_50000';
%bunName='sigmoid_bw_fixbeta_10000';
%bunName='sigmoid_bw_proposal_50000_v6';
%bunName='sigmoid_bw_proposal_25000';
%bunName='sigmoid_bw_proposal_5000';
%bunName = 'sigmoid_fw_proposal_5000';
% Nicolas's data. Has almost 30000 pairs.
%bunName=sprintf('nicolas_sigmoid_bw');
%bunName=sprintf('nicolas_sigmoid_fw');
%bunName=sprintf('simplegauss_d1_bw_proposal_30000' );
%bunName=sprintf('simplegauss_d1_fw_proposal_30000' );
bundle=se.loadBundle(bunName);

%n=5000;
%n=25000;
%[trBundle, teBundle] = bundle.partitionTrainTest(3000, 2000);
%[trBundle, teBundle] = bundle.partitionTrainTest(6000, 4000);
[trBundle, teBundle] = bundle.partitionTrainTest(3000, 1000);
%[trBundle, teBundle] = bundle.partitionTrainTest(10000, 10000);
%[trBundle, teBundle] = bundle.partitionTrainTest(40000, 10000);
%[trBundle, teBundle] = bundle.partitionTrainTest(500, 300);

%---------- options -----------
Xtr = trBundle.getInputTensorInstances();

out_msg_distbuilder = DNormalLogVarBuilder();

Ytr = out_msg_distbuilder.getStat(trBundle.getOutBundle());
% consider only means ?
Ytr = Ytr(1, :);
assert(isnumeric(Ytr));
mp = MatchingPursuit(Xtr, Ytr);

% median factors 
%medf = [1/10, 1/5, 1, 5, 10];
medf = [ 1/5, 1, 5  ];
mp_reg = 1e-2;
zfe = MVParamExtractor();
%xfe = NatParamExtractor();
xfe = MLogVParamExtractor();
% in order of variables p(z | x)
%fe = StackFeatureExtractor(zfe, xfe);
fc_candidates = getLaplaceFCCandidates(trBundle, zfe, xfe, medf);
% limit fc_candidates 
c = length(fc_candidates);
J = randperm(c, min(c, 100));
fc_candidates = fc_candidates(J);
display(sprintf('Totally %d function class candidates.', length(fc_candidates)));

od=mp.getOptionsDescription();
display(' MatchingPursuit options: ');
od.show();

% set my options
mp.opt('seed', seed);
mp.opt('mp_function_classes', fc_candidates);
mp.opt('mp_reg', mp_reg);
mp.opt('mp_subsample_proportion', 1.0);
mp.opt('mp_max_iters', 200);
mp.opt('mp_backfit_every', 1);
% start matching pursuit
mp.matchingPursuit();

n=length(trBundle)+length(teBundle);
iden=sprintf('mp_laplace_%s_%d.mat',  bunName, n);
fpath=Expr.scriptSavedFile(iden);

timeStamp=clock();
save(fpath, 'fc_candidates', 'mp', 'timeStamp', 'trBundle', 'teBundle');

rng(oldRng);
end

function candidates=getLaplaceFCCandidates(trBundle, zfe, xfe, medf)
    % z = Beta
    z = trBundle.getInputBundle(1);
    z = DistArray(z);
    x = trBundle.getInputBundle(2);
    x = DistArray(x);
    tensorTr = trBundle.getInputTensorInstances();
    n = length(x);
    % subsample
    subI = randperm(n, min(1e4, n));
    fz = zfe.extractFeatures(z(subI));
    fx = xfe.extractFeatures(x(subI));
    % for each feature dimension, find the median 
    zmeds = zeros(size(fz, 1), 1);
    xmeds = zeros(size(fx, 1), 1);
    for zi=1:length(zmeds)
        zmeds(zi) = meddistance(fz(zi, :));
    end
    for xi=1:length(xmeds)
        xmeds(xi) = meddistance(fx(xi, :));
    end
    assert(all(zmeds > 0));
    assert(all(xmeds > 0));
    % total number of candidats = len(medf)^totalDim
    allMeds = [zmeds(:)', xmeds(:)'];
    totalDim = length(allMeds);
    totalComb = length(medf)^totalDim;
    candidates = cell(1, totalComb);
    % temporary vector containing indices
    % Total combinations can be huge ! Be careful. Exponential in the 
    % number of inputs
    I = cell(1, totalDim);
    fe = StackFeatureExtractor(zfe, xfe);
    for ci=1:totalComb
        [I{:}] = ind2sub( length(medf)*ones(1, totalDim), ci);
        II=cell2mat(I);
        kerWidths= medf(II).*allMeds ;
        kerzWidths = kerWidths(1:size(fz, 1));
        kerxWidths = kerWidths( (size(fz, 1)+1):end);
        widths = [kerzWidths(:); kerxWidths(:)];
        centerInstances = tensorTr.instances(subI);
        candidates{ci} = KLaplaceFC(widths, fe, centerInstances, tensorTr );
    end
end

