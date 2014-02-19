function ds = getDescriptiveStatistics(x, param)
% getDescriptiveStatistics - calculates key values of the descriptive
% statistics
%
% function ds = getDescriptiveStatistics(x, param)
%
% Authors:                                                 in November 2010
%   (c) Matthias Chung (e-mail: mc85@txstate.edu)          
%       
% MATLAB Version: 7.10.0.499 (R2010a)
%
% Description:
%   This function gets the key values for a descriptive statistics such as 
%   mean, std, mode, median, quartiles and percentiles. This includes the 
%   five- and seven-number summary. Input x must be a vector or a matrix
%   where each column of x is regarded as a data set if x is a matrix.
%   Input param is an optional cell list of parameters, e.g., 
%   param = {'whisker', 2} setting whisker = 2. Multiple user
%   defined options are separated by semi-colon, e.g., param = {'whisker',
%   2; 'qmethod','-mean'}.
%   
% Input arguments:
%   x            - data x (each column is regarded as a data set)
%   #param       
%    whisker     - define region of outliers [1.5]
%    qmethod     - quartile method inclusive or exclusive mean [{'+mean'} | '-mean']
%    pmethod     - percentile method ['nearestRank' | {'interpolation'} | 'excel' | 'nist']
%    percent     - vector of percentages where percentiles are evaluated [2 9 25 50 75 91 98]
%
% Output arguments:
%   ds.          - structure of descriptive statistics
%     mean       - mean of x
%     sigma      - standard deviation of x
%     mode       - mode of x
%     median     - median of x
%     min        - minimal datum of x
%     max        - maximal datum of x
%     qmethod    - method to calculate quartiles [{'+mean'} | '-mean']
%     quartile   - vector of quartiles 25, 50, 75 percent of x
%     iqr        - interquartile range of x
%     sir        - semi interquartile range of x
%     whisker    - whisker outliers >0 [default 1.5]
%     nOutliers  - number of outliers of x
%     iOutliers  - interval excluding the outliers 
%     q1Outliers - list of extreme outliers below quartile 1
%     q3Outliers - list of extreme outliers above quartile 3
%     pmethod    - method to calculate the percentiles
%     percent    - list of percentages of calculated percentiles
%     percentile - list of percentiles for each sample set of x
%
% Examples:
%   ds = getDescriptiveStatistics(randn(1000,10))
%   ds = getDescriptiveStatistics(rand(1000,2),{'whisker', 2; 'qmethod','-mean'})
%   ds = getDescriptiveStatistics(rand(1000,2),{'percent', [25 50 75];'pmethod','nearestRank'})
%
% References:
%   [1] http://en.wikipedia.org/wiki/Descriptive_statistics

% initialize default options of function
whisker = 1.5; % default of 1.5 corresponds to approximately +/-2.7 sigma and 99.3 coverage if the data are normally distributed.
qmethod = '+mean'; % method to calculate quartiles (without '-mean' or inclusive mean '+mean')
pmethod = 'interpolation'; % method to calculate the percentiles
percent = [2 9 25 50 75 91 98]; % percentage where to calculate the percentiles.

% rewrite default parameters if needed
if nargin > 1, for j = 1:size(param,1), eval([param{j,1},'= param{j,2};']); end, end

% size of x
[n m] = size(x); if n == 1, x = x(:); n = length(x); m = 1; end

% compute mean
ds.mean = mean(x);

% compute standard deviation
ds.sigma = std(x);

% compute the mode
ds.mode = mode(x);

% compute the median
ds.median = median(x);

% compute the minimal datum
ds.min = min(x);

% compute the maximal datum
ds.max = max(x);

% initialize
ds.qmethod    = qmethod;
ds.quartile   = zeros(3,m); 
ds.iqr        = zeros(1,m); 
ds.sir        = zeros(1,m); 
ds.whisker    = whisker;  
ds.nOutliers  = zeros(1,m);
ds.iOutliers  = zeros(2,m);
ds.q1Outliers = {};
ds.q3Outliers = {};
ds.pmethod    = pmethod;
ds.percent    = percent;
ds.percentile = zeros(length(percent),m);

% compute 50th percentile (second quartile)
ds.quartile(2,:) = ds.median;

for j = 1:m
  
   y = sort(x(:,j));
  
   switch qmethod
     case '-mean'
       % compute 25th percentile (first quartile)
       ds.quartile(1,j) = median(y(y<ds.median(j)));
       
       % compute 75th percentile (third quartile)
       ds.quartile(3,j) = median(y(y>ds.median(j)));
       
     case '+mean'
       % compute 25th percentile (first quartile)
       ds.quartile(1,j) = median(y(y<=ds.median(j)));
       
       % compute 75th percentile (third quartile)
       ds.quartile(3,j) = median(y(y>=ds.median(j)));
       
     otherwise
       error('No quartile method set.')
   end
  
  % compute interquartile range
  ds.iqr(j) = ds.quartile(3,j) - ds.quartile(1,j);
  
  % compute semi interquartile range
  ds.sir(j) = ds.iqr(j)/2;
  
  % determine extreme q1 outliers (i.e., x < q1 - whisker (q3-q1))
  ds.iOutliers(1,j) = ds.quartile(1,j) - ds.whisker*ds.iqr(j);
  idx = find( y < ds.iOutliers(1,j) ); 
  if ~isempty(idx)
    ds.q1Outliers{j} = y(idx);
  else
    ds.q1Outliers{j} = [];
  end
  
  % determine extreme 3q outliers (i.e., x > q3 + whisker (q3-q1))
  ds.iOutliers(2,j) = ds.quartile(3,j) + ds.whisker*ds.iqr(j);
  idx = find( y > ds.iOutliers(2,j) );
  if ~isempty(idx)
    ds.q3Outliers{j} = y(idx);
  else
    ds.q3Outliers{j} = [];
  end
  
  % compute total number of outliers
  ds.nOutliers(j) = length(ds.q1Outliers{j})+length(ds.q3Outliers{j});
  
  % calculate percentiles
  for k = 1:length(percent)
    switch pmethod
      
      case 'nearestRank'
        % calculate rank and percentile
        r = round(percent(k)/100*(n+0.5));
        if r<1, r = 1; elseif r>n, r = n; end
        ds.percentile(k,j) = y(r);
        
      case 'interpolation'
        ds.percentile(k,j) = interp1([0, 100/n*((1:n)-0.5), 100], [y(1), y', y(end)], percent(k), 'linear');
        
      case 'weighted'
        error('Weighted percentile method not yet implemented.')
        
      case 'excel'
        in = percent(k)/100*(n-1)+1; K = floor(in); D = in - K;
        if K == 0
          ds.percentile(k,j) = y(1);
        elseif K == n
          ds.percentile(k,j) = y(n);
        else
          ds.percentile(k,j) = y(K) + D*(y(K+1) - y(K));
        end        
        
      case 'nist'
        in = percent(k)/100*(n+1); K = floor(in); D = in - K;
        if K == 0
          ds.percentile(k,j) = y(1);
        elseif K == n
          ds.percentile(k,j) = y(n);
        else
          ds.percentile(k,j) = y(K) + D*(y(K+1) - y(K));
        end
        
      otherwise
        error('No percentile method set.')
    end
  end
  
end

return