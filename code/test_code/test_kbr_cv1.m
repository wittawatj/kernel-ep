% function  test_kbr_cv1( )
% 
n = 400;

% p(y)
my = 0;
vy = 1;
Y = randn(1, n)*sqrt(vy) + my;

% uniform p(y)
% Y = rand(1, n)*8 -4;

% p(x|y)
a = -2;
b = 1;
vx = 0.5;
X = randn(1, n)*sqrt(vx) + a*Y + b;

% expected p(y|x)
pyv = 1/(1/vy + a*a/vx);
% Pym = pyv*(a/vx*(X - b) + my/vy );

% want to infer C_{y|x}.
o = [];
% C = kbr_cv1(X, Y,  op);
% C = cond_embed_cv1(X, Y, op);
% C = cond_embed_cv1(Y, X, op);

% both seem to work
[Op, CVLog] = CondOp1.kbr_operator(X, Y, o);
% [Op, CVLog] = CondOp1.learn_operator(X, Y, o);
skx = CVLog.bxw * CVLog.medx; %bxw = best Gaussian width for x

% print
fprintf('p(y) = N(%.2g, %.2g) \n', my ,vy);
fprintf('p(x|y) = N(%.2gy + %.2g, %.2g) \n', a, b, vx );

% test
Xtest = -6:6;
% expected effective X range roughly: -3:3
Eym = pyv*(a/vx*(Xtest - b) + my/vy );
for i=1:length(Xtest)
    xi = Xtest(i);
%     mxi_f = DistNormal(xi, 0.1);
    mxi_f = PointMass(xi);
    mfi_y = Op.apply_bp( mxi_f);
    fprintf('xi = %.2g, est. q(y|x) = N(%.2g, %.2g), true p(y|x) = N(%.2g, %.2g) \n', ...
        xi, mfi_y.mean, mfi_y.variance, Eym(i), pyv );
end

% end

