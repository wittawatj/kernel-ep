function t_clutter_kmv()
% Run clutter_kmv for multiple times, record inferred means and variances.
%

% How many runs
runs = 20;
op.t_clutter_kmv_path = 'saved/t_clutter_kmv_results.mat';

rerun = true;
if ~exist(op.t_clutter_kmv_path , 'file') || rerun
    
    % options
    op.clutter_theta_mean = 3;
    op.ep_iters = 10;
    op.observed_size = 500;
    % Each different seed will affect the observed values in EP. Not the
    % model because the model is not retrained.
    op.retrain_clutter_model = false;
    msetting.multicoreDir = myProcessOptions(op, 'multicoreDir', ...
        '/nfs/nhome/live/wittawat/SHARE/gatsby/research/code/tmp');
    
    % resultCell is a cell array of [R, op]. R contains the result of EP
    resultCell = startmulticoremaster( @(s)(fRunSeed(s, op)), num2cell(1:runs), msetting );
    
    % gather all inferred posterior q
    Q = DistNormal.empty(0, 1);
    RR = cell(1, length(resultCell));
    for i=1:length(resultCell)
        R = resultCell{i};
        Q(i) = R.q;
        RR{i} = R;
        op = dealstruct(R.op, op);
    end
    
    save(op.t_clutter_kmv_path, 'Q', 'op', 'RR');    
else
    % results exist, Load them
    load(op.t_clutter_kmv_path );
end

plotInferedParams(Q);

end


function [R] =fRunSeed(seed, op)
%
op.seed =seed;
[R] = clutter_kmv( op);
end

function plotInferedParams(Q)
% Q is an array of inferred DistNormal's from different seeds.
% means resulted from all seeds
Means = [Q.mean];
Vars = [Q.variance];
% scatter plot

scatter(Means, Vars);

end