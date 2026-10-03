function results = simulate_DGP2()
% SIMULATE_DGP2_DCOV_ORDERSTAT
% DGP 2: scalar local alternatives, proposed dCov test.
%
% X ~ Unif[-1,1]
% kappa(x) = exp(x/2)
% g(x) = sqrt(3)*x
%
% For each (n,k,c),
%   alpha_{n,c}(x) = 2 + (c/sqrt(k))*g(x),
%   Y = {kappa(X)/U}^{1/alpha_{n,c}(X)},  U ~ Unif(0,1).
%
% Thus c=0 is an exact-Pareto null with covariate-dependent tail frequency,
% while c>0 gives a k^{-1/2} local departure from tail-index homogeneity.
%
% Order-statistic implementation:
%   w_hat = Y_(n-k:n), so the largest k observations are selected.
%   Z_i^* = log(Y_i/w_hat).
% The numerical threshold cancels from pairwise log-exceedance distances.
%
% Settings requested for the current development run:
%   MC = 1000
%   Rperm = 299
%
% k-values:
%   n=2000: k=100,50
%   n=5000: k=200,100
%
% c-values:
%   c = 0,1,2,4
%
% Outputs:
%   DGP2_dcov_orderstat_summary.xlsx
%   DGP2_dcov_orderstat_results.csv
%   DGP2_dcov_orderstat_replications.mat

%% User settings
numWorkers = 48;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', numWorkers);
elseif pool.NumWorkers ~= numWorkers
    delete(pool);
    parpool('local', numWorkers);
end

baseSeed = 20260914;

nList = [2000, 5000];
kByN  = [100,  50; ...
         200, 100];
cList = [0, 1, 2, 4];

MC    = 1000;
Rperm = 299;
level = 0.05;

fprintf('DGP 2: proposed dCov test with order-statistic threshold\n');
fprintf('alpha_{n,c}(x) = 2 + c*sqrt(3)*x/sqrt(k)\n');
fprintf('MC = %d, permutations = %d\n\n', MC, Rperm);

%% Storage
nDesigns = numel(nList) * size(kByN,2) * numel(cList);

N_col       = zeros(nDesigns,1);
K_col       = zeros(nDesigns,1);
C_col       = zeros(nDesigns,1);
Reject_col  = zeros(nDesigns,1);
MCSE_col    = zeros(nDesigns,1);
AvgW_col    = zeros(nDesigns,1);
SdW_col     = zeros(nDesigns,1);
AlphaMin_col = zeros(nDesigns,1);
AlphaMax_col = zeros(nDesigns,1);
Perm_col    = Rperm * ones(nDesigns,1);
Level_col   = level * ones(nDesigns,1);

replications = cell(nDesigns,1);

row = 0;
tic;

for in = 1:numel(nList)
    n = nList(in);

    for ik = 1:size(kByN,2)
        k = kByN(in,ik);

        for ic = 1:numel(cList)
            c = cList(ic);

            row = row + 1;
            designID = row;

            rejected = false(MC,1);
            pvalues  = NaN(MC,1);
            wHatRep  = NaN(MC,1);

            fprintf('Running n=%d, k=%d, c=%.1f ...\n', n, k, c);

            parfor rep = 1:MC
                rng(baseSeed + 100000*designID + rep, 'twister');

                % ----- Generate DGP 2 -----
                X = 2*rand(n,1) - 1;
                U = rand(n,1);
                U(U == 0) = realmin;

                kappa = exp(X/2);
                g = sqrt(3)*X;
                alpha = 2 + (c/sqrt(k))*g;

                if any(alpha <= 0)
                    error('DGP 2 generated a nonpositive tail exponent.');
                end

                Y = (kappa ./ U).^(1 ./ alpha);

                % ----- Select top k observations -----
                [Ydesc, order] = sort(Y, 'descend');
                wHat = Ydesc(k+1);
                tailIdx = order(1:k);

                Xe = X(tailIdx);
                Ze = log(Y(tailIdx) ./ wHat);

                % ----- dCov permutation test -----
                pval = dcov_permutation_pvalue(Xe, Ze, Rperm);

                pvalues(rep) = pval;
                rejected(rep) = (pval <= level);
                wHatRep(rep) = wHat;
            end

            rejectRate = mean(rejected);

            N_col(row)       = n;
            K_col(row)       = k;
            C_col(row)       = c;
            Reject_col(row)  = rejectRate;
            MCSE_col(row)    = sqrt(rejectRate*(1-rejectRate)/MC);
            AvgW_col(row)    = mean(wHatRep);
            SdW_col(row)     = std(wHatRep);
            AlphaMin_col(row)= 2 - c*sqrt(3)/sqrt(k);
            AlphaMax_col(row)= 2 + c*sqrt(3)/sqrt(k);

            replications{row} = struct( ...
                'n', n, 'k', k, 'c', c, ...
                'pvalue', pvalues, ...
                'reject', rejected, ...
                'w_hat', wHatRep);

            fprintf('  rejection rate = %.4f, MCSE = %.4f\n', ...
                Reject_col(row), MCSE_col(row));
        end
    end
end

toc;

%% Save
results = table( ...
    N_col, K_col, C_col, AlphaMin_col, AlphaMax_col, ...
    Reject_col, MCSE_col, AvgW_col, SdW_col, Perm_col, Level_col, ...
    'VariableNames', { ...
        'n','k','c','alpha_min','alpha_max', ...
        'rejection_rate','mcse','mean_w_hat','sd_w_hat', ...
        'permutations','nominal_level'});

disp(results);

writetable(results, 'DGP2_dcov_orderstat_summary.xlsx', 'Sheet', 'Summary');
writetable(results, 'DGP2_dcov_orderstat_results.csv');

settings = table(MC, Rperm, level, numWorkers, baseSeed, ...
    'VariableNames', {'MC_replications','permutations','nominal_level', ...
                      'parallel_workers','base_seed'});
writetable(settings, 'DGP2_dcov_orderstat_summary.xlsx', 'Sheet', 'Settings');

save('DGP2_dcov_orderstat_replications.mat', ...
    'replications','results','settings','nList','kByN','cList', ...
    'MC','Rperm','level','numWorkers','baseSeed','-v7.3');

%% Plot power
figure;
hold on;
for in = 1:numel(nList)
    for ik = 1:size(kByN,2)
        n = nList(in);
        k = kByN(in,ik);
        rows = (results.n == n) & (results.k == k);
        plot(results.c(rows), results.rejection_rate(rows), '-o', ...
            'LineWidth',1.3, ...
            'DisplayName',sprintf('n=%d, k=%d',n,k));
    end
end
yline(level,'--','Nominal 5%','LineWidth',1.1);
xlabel('Local-alternative parameter c');
ylabel('Rejection probability');
title('DGP 2: dCov power');
legend('Location','best');
grid on;
hold off;

end


function pval = dcov_permutation_pvalue(X, Z, Rperm)

X = X(:);
Z = Z(:);
m = numel(X);

A = centered_distance_matrix(X);
B = centered_distance_matrix(Z);

Tobs = sum(A .* B, 'all') / (m^2);

nGreaterEqual = 0;
for r = 1:Rperm
    pi = randperm(m);
    Bpi = B(pi,pi);
    Tperm = sum(A .* Bpi, 'all') / (m^2);
    nGreaterEqual = nGreaterEqual + (Tperm >= Tobs);
end

pval = (1 + nGreaterEqual)/(Rperm + 1);

end


function A = centered_distance_matrix(V)

V = V(:);
D = abs(V - V.');
rowMean = mean(D,2);
colMean = mean(D,1);
grandMean = mean(D,'all');
A = D - rowMean - colMean + grandMean;

end
