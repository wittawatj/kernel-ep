function [ ] = test_cmaes(  )
%TEST_CMAES test cmaes package
%

%option:
%%
%                  StopFitness: '-Inf % stop if f(xmin) < stopfitness, minimization'
%                  MaxFunEvals: 'Inf  % maximal number of fevals'
%                      MaxIter: '1e3*(N+5)^2/sqrt(popsize) % maximal number of iterations'
%                 StopFunEvals: 'Inf  % stop after resp. evaluation, possibly resume later'
%                     StopIter: 'Inf  % stop after resp. iteration, possibly resume later'
%                         TolX: '1e-11*max(insigma) % stop if x-change smaller TolX'
%                       TolUpX: '1e3*max(insigma) % stop if x-changes larger TolUpX'
%                       TolFun: '1e-12 % stop if fun-changes smaller TolFun'
%                   TolHistFun: '1e-13 % stop if back fun-changes smaller TolHistFun'
%             StopOnStagnation: 'on  % stop when fitness stagnates for a long time'
%               StopOnWarnings: 'yes  % 'no'=='off'==0, 'on'=='yes'==1 '
%    StopOnEqualFunctionValues: '2 + N/3  % number of iterations'
%                DiffMaxChange: 'Inf  % maximal variable change(s), can be Nx1-vector'
%                DiffMinChange: '0    % minimal variable change(s), can be Nx1-vector'
%    WarnOnEqualFunctionValues: 'yes  % 'no'=='off'==0, 'on'=='yes'==1 '
%                      LBounds: '-Inf % lower bounds, scalar or Nx1-vector'
%                      UBounds: 'Inf  % upper bounds, scalar or Nx1-vector'
%                 EvalParallel: 'no   % objective function FUN accepts NxM matrix, with M>1?'
%                 EvalInitialX: 'yes  % evaluation of initial solution'
%                     Restarts: '0    % number of restarts '
%                   IncPopSize: '2    % multiplier for population size before each restart'
%                      PopSize: '(4 + floor(3*log(N)))  % population size, lambda'
%                 ParentNumber: 'floor(popsize/2)       % AKA mu, popsize equals lambda'
%         RecombinationWeights: 'superlinear decrease % or linear, or equal'
%                 DiagonalOnly: '0*(1+100*N/sqrt(popsize))+(N>=1000)  % C is diagonal for given iterations, 1==always'
%                        Noise: [1x1 struct]
%                          CMA: [1x1 struct]
%                       Resume: 'no   % resume former run from SaveFile'
%                      Science: 'on  % off==do some additional (minor) problem capturing, NOT IN USE'
%                  ReadSignals: 'on  % from file signals.par for termination, yet a stumb'
%                         Seed: 'sum(100*clock)  % evaluated if it is a string'
%                    DispFinal: 'on   % display messages like initial and final message'
%                   DispModulo: '100  % [0:Inf], disp messages after every i-th iteration'
%                SaveVariables: 'on   % [on|final|off][-v6] save variables to .mat file'
%                 SaveFilename: 'variablescmaes.mat  % save all variables, see SaveVariables'
%                    LogModulo: '1    % [0:Inf] if >1 record data less frequently after gen=100'
%                      LogTime: '25   % [0:100] max. percentage of time for recording data'
%            LogFilenamePrefix: 'outcmaes  % files for output data'
%                      LogPlot: 'off    % plot while running using output data files'
%                     UserData: 'for saving data/comments associated with the run'
%                     UserDat2: ''
opt = struct();
opt.StopFitness = 0;
opt.MaxFunEvals = 1000;
%opt.LBounds = 0;
%opt.UBounds = 1000;
opt.DiffMinChange = 1e-3;
opt.TolX = 1e-3;
opt.TolFun = 1e-3;
opt.Noise.on = true;
opt.DispModulo = 10;
opt.LogModulo = 0;
opt.SaveVariables = 'off';

x0 = randn(5, 1)*20;
xmin = cmaes(@norm_func, x0, 10, opt)

%x0 = [-3, 5]';
%xmin = cmaes(@rosenbrock, x0, 300)


end

function f=norm_func(x)
    A = randn(500, 2000)*randn(2000, 500);
    xmin = [2, -1, 3, -10, 100]';
    f = norm(x - xmin);
end

function f = rosenbrock(x)
    f = 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
end

