% Convert Nicolas's data to MsgBundle and save 
%

load('saved/external/sigmoidFactor_fw.mat');

% xtrain 4x.. (mean, log-precision, log alpha, log beta)
gaussMeans=xtrain(1, :);
gaussVars=1./exp(xtrain(2, :));
assert(all(gaussVars>0));
alphas=exp(xtrain(3, :));
assert(all(alphas>0));
betas=exp(xtrain(4, :));
assert(all(betas>0));

inDistArray1=DistBeta(alphas, betas);
inDistArray1=DistArray(inDistArray1);
inDistArray2=DistNormal(gaussMeans, gaussVars);
inDistArray2=DistArray(inDistArray2);

% ytrain has alpha, beta
alphasOut=ytrain(1, :);
assert(all(alphasOut>0));
betasOut=ytrain(2, :);
assert(all(betasOut>0));
outDistArray=DistBeta(alphasOut, betasOut);
outDistArray=DistArray(outDistArray);

% order input as (beta, normal)
se=BundleSerializer();
fwBundle=DefaultMsgBundle(outDistArray, inDistArray1, inDistArray2);
se.saveBundle(fwBundle, 'nicolas_sigmoid_fw');


% xtrain the same 
% ytrain contains mean, variance 
load('saved/external/sigmoidFactor_bw.mat');
outMeans=ytrain(1, :);
outVars=ytrain(2, :);
assert(all(outVars>0));
bwOutDistArray=DistNormal(outMeans, outVars);
bwOutDistArray=DistArray(bwOutDistArray);
se=BundleSerializer();
bwBundle=DefaultMsgBundle(bwOutDistArray, inDistArray1, inDistArray2);
se.saveBundle(bwBundle, 'nicolas_sigmoid_bw');



