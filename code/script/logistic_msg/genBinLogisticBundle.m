function [ ] = genBinLogisticBundle(s, bundleName )
%GENBINLOGISTICBUNDLE Generate a MsgBundle from an exported .mat file from 
%Infer.NET containing recorded incoming/outgoing messages during EP inference 
%on a binary logistic regression model.
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

inBetas = DistBeta(s.inBetaA', s.inBetaB');
inBetas = DistArray(inBetas);

inNormals = DistNormal(s.inNormalMeans', s.inNormalVariances');
inNormals = DistArray(inNormals);

outNormals = DistNormal(s.outNormalMeans', s.outNormalVariances');
outNormals = DistArray(outNormals);


bundle = DefaultMsgBundle(outNormals, inBetas, inNormals);
serializer = BundleSerializer();
serializer.saveBundle(bundle, bundleName);

end

