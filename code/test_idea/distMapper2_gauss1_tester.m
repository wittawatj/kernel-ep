function  [hmean, hvar, hkl]=distMapper2_gauss1_tester( mapper,  X1, X2, Out)
%DISTMAPPER2_GAUSS1_TESTER A generic tester for DistMapper2.
% 
%   Given a dataset and a mapper, compare the mapped output 
%   out' = mapper(x1, x2) to the ground truth in Out. Then plot.
% 
%   Only of Gaussian inputs and Gaussian output.
%
%   X1, X2, Out = array of DistNormal
%   Return handles to the plots.
% 
assert(isa(X1, 'DistNormal'));
assert(isa(X2, 'DistNormal'));
assert(isa(Out, 'DistNormal'));
assert(length(X1)==length(X2));
assert(length(X2)==length(Out));

% n test
nte = length(X1);
KL = zeros(1, nte);
OpMsgs = DistNormal.empty(0, nte);
% test the operator on the training set.
for i=1:nte
    x1 = X1(i);
    x2 = X2(i);
    q = mapper.mapDist2(x1, x2);
    
    true_toTq = Out(i);
    if true_toTq.isproper() && q.isproper()
        % compare mfi_z to the one from training set
        kl = kl_gauss(true_toTq, q);
    else
        kl = nan();
    end
    KL(i) = kl;
    OpMsgs(i) = q;
end

% plot to compare training and output messages
% means
TrMeans = [Out.mean];
OpMeans = [OpMsgs.mean];
hmean=figure;
hold on
set(gca, 'fontsize', 20);
stem(TrMeans, 'or');
stem(OpMeans, 'ob');
plot( abs(TrMeans-OpMeans), '-k', 'LineWidth', 2);
xlabel('Message index');
ylabel('Gaussian mean');
% title(sprintf('Training size: %d', n));
legend('True means', 'Output means', 'abs. diff.');
grid on
hold off

% variance
TrVar = [Out.variance];
OpVar = [OpMsgs.variance];
hvar= figure;
hold on
set(gca, 'fontsize', 20);
stem(TrVar, 'or');
stem(OpVar, 'ob');
plot( abs(TrVar-OpVar), '-k', 'LineWidth', 2);
xlabel('Message index');
ylabel('Gaussian variance');
% title(sprintf('Training size: %d', n));
legend('True variance', 'Output variance', 'abs. diff.');
grid on
hold off

% plot KL
hkl=figure;
stem(KL);
set(gca, 'fontsize', 20);
title('KL error on training messages' );
xlabel('Messsage index');
ylabel('KL');


end

