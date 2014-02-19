function s = dealstruct(s1, s2)
%
% Merge fields in s2 (struct) into s1 (struct).
% If there are same fields, then values in s2 replace s1.
%
% Is there an easier way to do this in Matlab ??
%
if length(s1) > 1 || length(s2) > 1
    error('%s does not work for arrays.', mfilename);
end

s = s1;
fName = fieldnames(s2);
for i=1:length(fName)
    f = fName{i};
    s.(f) = s2.(f);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end