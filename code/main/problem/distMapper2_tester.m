function  [hmean, hvar, hhell]=distMapper2_tester( mapper,  X1, X2, Out)
%DISTMAPPER2_TESTER A generic tester for DistMapper2.
% 
%   Given a dataset and a mapper, compare the mapped output 
%   out' = mapper(x1, x2) to the ground truth in Out. Then plot.
%
%   X1, X2 = array such that each element is accepted by the mapper
%   Out = array of Distribution
%   Return handles to the plots.
% 

assert(length(X1)==length(X2));
assert(length(X2)==length(Out));

% n test
nte = length(X1);
Helling = zeros(1, nte);
distBuilder = Out(1).getDistBuilder();
OpMsgs = distBuilder.empty(0, nte);
% test the operator on the training set.
for i=1:nte
    x1 = X1(i);
    x2 = X2(i);
    q = mapper.mapDist2(x1, x2);
    
    true_toTq = Out(i);
    if true_toTq.isProper() && q.isProper()
        % compare mfi_z to the one from training set
        hl = true_toTq.distHellinger(q);
    else
        hl = nan();
    end
    Helling(i) = hl;
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
ylabel(sprintf('Mean of %s', class(Out(1)) ));
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
ylabel(sprintf('Variance of %s', class(Out(1)) ));
% title(sprintf('Training size: %d', n));
legend('True variance', 'Output variance', 'abs. diff.');
grid on
hold off

% plot Hellinger distances 
hhell=figure;
hold on
stem(Helling);
set(gca, 'fontsize', 20);
title(sprintf('Hellinger distance on %s', class(Out(1))) );
xlabel('Messsage index');
ylabel('Hellinger distance');
ylim([0, 1]);
grid on
hold off

end

