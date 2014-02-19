function str = kerfunc2str(kerFunc)
% 
% Kernel function to string
%

[ fname, param] = kerfuncexplode( func2str(kerFunc) );
str = sprintf('%s(%.4g)', fname, str2double(param));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

