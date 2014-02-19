function [ BestMat ] = dmetest( DME, alpha )
%
% Perform paired T-test on (dataset x method x Err array) matrix DME.
% Return a binary matrix indicating which methods are equivalent 
% (no significant difference) to the best method, for each dataset.
%
% (alpha) is the significance level.
%

% Find the best method(s) for each dataset
[da, me, er] = size(DME);

% Binary matrix telling which method is(are) the best for each dataset.
BestMat = false(da, me);
for di=1:da
    
    ME = shiftdim(DME(di, :, :), 1); % methods x errors
    meanErr = nanmean(ME, 2);
    stdErr = nanstd(ME, 0 , 2);
    
    % First sort Errs based on mean , std in ascending order
    [SortedME, I] = sortrows([ meanErr, stdErr] , [1 2]);
    
    % index of the best method (row index)
    bestmethodj = I(1);
    BestErr = ME(bestmethodj, :);
    BestMat(di, bestmethodj) = true; 
    
    % Compare the first one with the rest to find equivalence(s)
    % Test the best method with all others 
    
%     H = ttest(repmat(BestErr', 1, me), ME', alpha, 'left');
    H = ttest2(repmat(BestErr', 1, me), ME', alpha, 'left','unequal');
    
    % ttest(X,Y)=NaN when X and Y are very similar.
    H(isnan(H)) = true;
    
    BestMat(di, ~H) = true;
    
%     for j=1:da
%             
%         %[h,p]= signrank(BestErr, ME(j,:) );
% 
%         % Use one-sided paired t-test 
%         [h, p]= ttest(BestErr, ME(j,:), alpha, 'left');
% 
%         if h == 0 % failure to reject the null hypothesis
%             BestMat(r, j) = true; % method j is also equivalent to the best
%         end
%         
%         
%     end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
