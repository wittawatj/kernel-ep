function exp_gamma()
% 
% Exponential likelihood, gamma prior
% 

% gamma prior parameters
alpha = 5;
beta = 1;
n = 800;
L = gamrnd(alpha, 1/beta, 1, n);

E = exprnd(1./L, 1, n);

hist(E, 20)

end
