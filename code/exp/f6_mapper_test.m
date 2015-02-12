function f6_mapper_test()
% A script to construct a DivDistMapperTester on loaded result of exp6. 
% This may also work with other experiments.
% A result file must have been loaded first.
%

% Expected variables in the global environment:
%
%  Name                     Size            Bytes  Class                   Attributes
%  out_msg_distbuilder      1x1               112  DNormalLogVarBuilder              
%  s                        1x1              4364  struct                            
%  teBundle                 1x1               112  DefaultMsgBundle                  
%  timeStamp                1x6                48  double                            
%  trBundle                 1x1               112  DefaultMsgBundle                  
%
% s = 
%      learner_class: 'RFGJointKGGLearner'
%    learner_options: [1x1 Options]
%        dist_mapper: [1x1 UAwareGenericMapper]
%        learner_log: [1x1 struct]
%             commit: '59823d15'
%          timeStamp: [2015 2 12 12 1 37.2040]
%

seed = 3;
oldRng = rng();
rng(seed);

% Access base environment 
s = evalin('base', 's');
out_msg_distbuilder = evalin('base', 'out_msg_distbuilder');
teBundle = evalin('base', 'teBundle');
trBundle = evalin('base', 'trBundle');

%divTester = DivDistMapperTester(s.dist_mapper);
%divTester.testDistMapper(teBundle);

% ---- Plot log KL vs. predictive variance ---

dm = s.dist_mapper;
assert(isa(teBundle, 'MsgBundle'));
teOutDa= dm.mapMsgBundle(teBundle);
assert(isa(teOutDa, 'DistArray'));
teTrueOutDa=teBundle.getOutBundle();
assert(isa(teTrueOutDa, 'DistArray'));
trOutDa = dm.mapMsgBundle(trBundle);
trTrueOutDa = trBundle.getOutBundle();

divTester = DivDistMapperTester(dm);
% KL divergences. May contain nan.
teDivs = divTester.getDivergence(teOutDa, teTrueOutDa);
trDivs = divTester.getDivergence(trOutDa, trTrueOutDa);

% uncertainties 
teU = dm.estimateUncertaintyMsgBundle(teBundle);
trU = dm.estimateUncertaintyMsgBundle(trBundle);

% exclude KL div. values which are nan. (improper output messages)
teI = isfinite(teDivs) & isreal(teDivs) ;
trI = isfinite(trDivs) & isreal(trDivs) ;

plotUncertainty( log(trDivs(trI)), log(trU(1, trI)), log(teDivs(teI)), ...
    log(teU(1, teI)), 'Log KL error', 'Log predictive variance', ...
    sprintf('Predicting output 1. %s', out_msg_distbuilder.shortSummary()) );

plotUncertainty( log(trDivs(trI)), log(trU(2, trI)), log(teDivs(teI)), ...
    log(teU(2, teI)), 'Log KL error', 'Log predictive variance', ...
    sprintf('Predicting output 2. %s', out_msg_distbuilder.shortSummary()) );

rng(oldRng);
end


function plotUncertainty(trDivs, trU, teDivs, teU, xaxis, yaxis, titleText)

    % To prevent cluttered results
    subsamples = 700;
    trN = length(trDivs);
    teN = length(teDivs);
    trSub = randperm(trN, min(subsamples, trN) );
    teSub = randperm(teN, min(subsamples, teN) );
    figure
    hold on 

    plot(teDivs(teSub), teU(teSub), 'xr', 'LineWidth', 1 );
    plot(trDivs(trSub), trU(trSub), '*k', 'LineWidth', 1, 'MarkerSize', 4 );
    set(gca, 'FontSize', 24);
    xlabel(xaxis);
    ylabel(yaxis);
    title(titleText);
    %grid on
    legend( 'Test set', 'Training set');
    hold off 

end
