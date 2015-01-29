function [ s ] = funcs_logistic_msgs(  )
%FUNCS_LOGISTIC_MSGS A set of functions to implement necessary operations to 
%bring object from/to Infer.NET
%

    % Return a struct containing functions for exp3
    s=struct();
    s.convertBinLogFWBundle = @convertBinLogFWBundle;
    s.convertBinLogBWBundle = @convertBinLogBWBundle;
    s.genBinLogFWBundle = @genBinLogFWBundle;
    s.genBinLogBWBundle = @genBinLogBWBundle;

end

function bundle = convertBinLogFWBundle(bundleName)
    % Read, convert and write to saved/bundle/ folder.
    scriptFol = Global.getScriptFolder();
    fromInferFile = fullfile(scriptFol, 'logistic_msg', 'infer.net_saved', bundleName);
    s = load(fromInferFile);
    bundle = genBinLogFWBundle(s, bundleName);
end

function bundle=genBinLogFWBundle(s, bundleName )
% Generate a MsgBundle from an exported .mat file from 
%Infer.NET containing recorded incoming/outgoing messages during EP inference 
%on a binary logistic regression model. Forward direction i.e. to Beta variable.
%
%  Input:
%  - s is the struct obtained by s = load('...')
%
% Example of what is inside the .mat file 
%
%%  Name                            Size            Bytes  Class      Attributes
%  X                             400x1             76800  cell                 
%  Y                             400x1               400  logical              
%  inBetaA                       400x1              3200  double               
%  inBetaB                       400x1              3200  double               
%  inNormalMeans                 400x1              3200  double               
%  inNormalVariances             400x1              3200  double               
%  outBetaA                      400x1              3200  double               
%  outBetaB                      400x1              3200  double               
%  regression_input_dim            1x1                 8  double               
%  regression_training_size        1x1                 8  double               
%  true_bias                       1x1                 8  double               
%  true_w                         10x1                80  double               


isProperFunc = @(d)d.isProper();
inBetas = DistBeta(s.inBetaA', s.inBetaB');
inBetasProper = arrayfun(isProperFunc, inBetas);

inNormals = DistNormal(s.inNormalMeans', s.inNormalVariances');
inNormalsProper = arrayfun(isProperFunc, inNormals);

outBetas = DistBeta(s.outBetaA', s.outBetaB');
outBetasProper = arrayfun(isProperFunc, outBetas);

% indices of proper messages
I = inBetasProper & inNormalsProper & outBetasProper;

inBetas = DistArray(inBetas(I));
inNormals = DistArray(inNormals(I));
outBetas = DistArray(outBetas(I));

bundle = DefaultMsgBundle(outBetas, inBetas, inNormals);
serializer = BundleSerializer();
serializer.saveBundle(bundle, bundleName);

end

function bundle = convertBinLogBWBundle(bundleName)
    % Read, convert and write to saved/bundle/ folder.
    scriptFol = Global.getScriptFolder();
    fromInferFile = fullfile(scriptFol, 'logistic_msg', 'infer.net_saved', bundleName);
    s = load(fromInferFile);
    bundle = genBinLogBWBundle(s, bundleName);
end

function bundle=genBinLogBWBundle(s, bundleName )
% Generate a MsgBundle from an exported .mat file from 
%Infer.NET containing recorded incoming/outgoing messages during EP inference 
%on a binary logistic regression model. Backward direction i.e. to Gaussian 
% variable.
%
%  Input:
%  - s is the struct obtained by s = load('...')
%
% Example of what is inside the .mat file 
%
%  X                              300x1               57600  cell                 
%  Y                              300x1                 300  logical              
%  ans                              1x6300            50400  double               
%  inBetaA                       6300x1               50400  double               
%  inBetaB                       6300x1               50400  double               
%  inNormalMeans                 6300x1               50400  double               
%  inNormalVariances             6300x1               50400  double               
%  outNormalMeans                6300x1               50400  double               
%  outNormalVariances            6300x1               50400  double               
%  regression_input_dim             1x1                   8  double               
%  regression_training_size         1x1                   8  double               
%  true_bias                        1x1                   8  double               
%  true_w                          10x1                  80  double  
%

isProperFunc = @(d)d.isProper();
inBetas = DistBeta(s.inBetaA', s.inBetaB');
inBetasProper = arrayfun(isProperFunc, inBetas);

inNormals = DistNormal(s.inNormalMeans', s.inNormalVariances');
inNormalsProper = arrayfun(isProperFunc, inNormals);

outNormals = DistNormal(s.outNormalMeans', s.outNormalVariances');
outNormalsProper = arrayfun(isProperFunc, outNormals);

% indices of proper messages
I = inBetasProper & inNormalsProper & outNormalsProper;

inBetas = DistArray(inBetas(I));
inNormals = DistArray(inNormals(I));
outNormals = DistArray(outNormals(I));

bundle = DefaultMsgBundle(outNormals, inBetas, inNormals);
serializer = BundleSerializer();
serializer.saveBundle(bundle, bundleName);

end


%function genSigmoidFactorOperator()
%    % convenient method to generate a sigmoid FactorOperator from exp3 results.
%    fwFile='RFGMVMapperLearner_nicolas_sigmoid_fw_25000.mat';
%    bwFile='RFGMVMapperLearner_nicolas_sigmoid_bw_25000.mat';

%    foName='RFGMVMapperLearner_nicolas_sigmoid_25000';

%    summary=['sigmoid factor. p(x1|x2) where x1 is Beta and '...
%        'x2 is Normal. RFGMVMapperLearner_nicolas_sigmoid_25000'];
%    fo=getSigmoidFactorOperator(fwFile, bwFile, summary);
%    serializer=FactorOpSerializer();
%    % save operator 
%    serializer.saveFactorOperator(fo, foName);

%    % serialize operator for C#
%    serializer.serializeFactorOperator(fo, foName);

%end

%function fo=getSigmoidFactorOperator(fwFile, bwFile, summary)
%    % Construct a FactorOperator for sigmoid problem. 
%    % This requires loading from at least two files in exp3 results as 
%    % we need one DistMapper for forward and another for backward direction.
%    %
%    % fwFile = file name in exp3 folder 
%    % bwFile = 
%    %
    
%    fwPath=Expr.expSavedFile(3, fwFile);
%    % expect a struct s
%    load(fwPath);
%    fwResult=s;
%    assert(isa(fwResult.out_distarray(1), 'DistBeta'));

%    bwPath=Expr.expSavedFile(3, bwFile);
%    load(bwPath);
%    bwResult=s;
%    assert(isa(bwResult.out_distarray(1), 'DistNormal'));
%    %  learner_class: 'RFGMVMapperLearner'
%    %learner_options: [1x1 Options]
%    %    result_path: '/nfs/nhome/live/wittawat/SHARE/gatsby/research/code/saved/exp/exp3/RFGMVMapperLearner_nicolas_sigmoid_fw_25000.mat'
%    %    dist_mapper: [1x1 GenericMapper]
%    %    learner_log: [1x1 struct]
%    %     div_tester: [1x1 DivDistMapperTester]
%    %        helling: [1x5000 double]
%    %  out_distarray: [1x1 DistArray]
%    %     imp_tester: [1x1 ImproperDistMapperTester]
%    %        imp_out: [1x2358 DistBeta]
%    %         commit: 'b1535cc4'
%    %      timeStamp: [2014 7 27 17 30 47.8046]

%    distMapper1=fwResult.dist_mapper;
%    distMapper2=bwResult.dist_mapper;
%    fo=DefaultFactorOperator({distMapper1, distMapper2}, summary);
%end


