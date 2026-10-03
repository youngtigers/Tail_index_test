function results = simulate_DGP1_beta1_global()
% SIMULATE_DGP1_BETA1_GLOBAL
% Monte Carlo size experiment for DGP 1 (Hall-type null with beta = 1),
% using an INTERMEDIATE ORDER-STATISTIC threshold.
%
% DGP 1:
%   X ~ Unif[-1,1]
%   s(X) = exp(X/2)
%   Y = s(X)*(U^(-1/2)-1),  U ~ Unif(0,1)
%
% Hence
%   P(Y > y | X=x) = [1 + y/s(x)]^(-2)
%                    = s(x)^2*y^(-2)*{1 - 2*s(x)*y^(-1) + O(y^(-2))}.
%
% Thus alpha0 = 2 and the Hall second-order exponent is beta = 1,
% while both the leading tail-frequency term and the second-order
% coefficient vary with X.
%
% THRESHOLD CHOICE:
%   For each (n,k), let
%
%       w_hat = Y_(n-k:n),
%
%   so exactly the largest k observations satisfy Y_i > w_hat almost
%   surely under this continuous DGP. The dCov test is then applied to
%   those k observations.
%
%   We use the same k-values as in the revised simulation design:
%
%       n = 2000: k = 100, 50
%       n = 5000: k = 200, 100
%
%   These choices match the tail-sample sizes used in the de Haan-Zhou
%   benchmark in the paper.
%
% For selected observations,
%
%       Z_i^* = log(Y_i / w_hat).
%
% Since
%
%       |log(Y_i/w_hat)-log(Y_j/w_hat)|
%       = |log(Y_i)-log(Y_j)|,
%
% the numerical value of w_hat does not affect the dCov statistic after
% the top k observations have been selected. We nevertheless construct
% Z_i^* explicitly to match the notation in the paper.
%
% OUTPUT:
%   results : MATLAB table containing empirical rejection frequencies,
%             Monte Carlo standard errors, and realized threshold
%             summaries for each (n,k) design.
%
% The script writes:
%   DGP1_beta1_dcov_orderstat_summary.xlsx
%   DGP1_beta1_dcov_orderstat_results.csv
%   DGP1_beta1_dcov_orderstat_replications.mat
%
% NOTE:
%   This file revises the dCov implementation and threshold choice only.
%   The Wang-Li (2013) and Kinsvater-Fried (2017) comparison routines
%   should be added as separate, validated functions so their original
%   procedures are not inadvertently altered.

%% User settings

% Parallel computing
numWorkers = 8;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', numWorkers);
elseif pool.NumWorkers ~= numWorkers
    delete(pool);
    parpool('local', numWorkers);
end

baseSeed = 20260914;

nList = [2000, 5000];

% Rows correspond to nList.
% Larger and smaller tail samples, respectively.
kByN = [100,  50; ...
        200, 100];

MC    = 1000;     % final Monte Carlo replications
Rperm = 299;      % final random permutations
level = 0.05;     % nominal test size

fprintf('DGP 1: Hall-type null, alpha0 = 2, beta = 1\n');
fprintf('Order-statistic threshold: w_hat = Y_(n-k:n)\n');
for in = 1:numel(nList)
    fprintf('  n = %d: k = %d, %d\n', ...
        nList(in), kByN(in,1), kByN(in,2));
end
fprintf('\n');

%% Monte Carlo experiment

nDesigns = numel(nList) * size(kByN,2);

N_col        = zeros(nDesigns,1);
K_col        = zeros(nDesigns,1);
Kfrac_col    = zeros(nDesigns,1);
Reject_col   = zeros(nDesigns,1);
MCSE_col     = zeros(nDesigns,1);
AvgW_col     = zeros(nDesigns,1);
SdW_col      = zeros(nDesigns,1);
MedianW_col  = zeros(nDesigns,1);
Perm_col     = Rperm * ones(nDesigns,1);
Level_col    = level * ones(nDesigns,1);

replications = cell(nDesigns,1);

row = 0;
tic;

for in = 1:numel(nList)

    n = nList(in);

    for ik = 1:size(kByN,2)

        k = kByN(in,ik);
        designID = (in-1)*size(kByN,2) + ik;

        rejected = false(MC,1);
        pvalues  = NaN(MC,1);
        wHatRep  = NaN(MC,1);

        fprintf('Running n = %d, k = %d ...\n', n, k);

        parfor rep = 1:MC

            % Reproducible replication-specific seed.
            rng(baseSeed + 100000*designID + rep, 'twister');

            % -------------------------------------------------------------
            % Generate DGP 1
            % -------------------------------------------------------------
            X = 2*rand(n,1) - 1;
            U = rand(n,1);
            U(U == 0) = realmin;

            s = exp(X/2);
            Y = s .* (U.^(-1/2) - 1);

            % -------------------------------------------------------------
            % Intermediate order-statistic threshold
            %
            % Ascending notation:
            %   Y_(1:n) <= ... <= Y_(n:n)
            %
            % w_hat = Y_(n-k:n) is the (k+1)-th largest observation.
            % Therefore exactly the largest k observations exceed w_hat
            % almost surely because the DGP is continuous.
            % -------------------------------------------------------------
            [Ydesc, order] = sort(Y, 'descend');

            wHat = Ydesc(k+1);
            tailIdx = order(1:k);

            Xe = X(tailIdx);
            Ze = log(Y(tailIdx) ./ wHat);

            % Defensive check: the selected tail sample must contain k
            % observations exactly.
            if numel(Ze) ~= k
                error('Tail selection did not produce exactly k observations.');
            end

            % -------------------------------------------------------------
            % dCov permutation test
            % -------------------------------------------------------------
            pval = dcov_permutation_pvalue(Xe, Ze, Rperm);

            pvalues(rep) = pval;
            rejected(rep) = (pval <= level);
            wHatRep(rep) = wHat;
        end

        row = row + 1;

        rejectRate = mean(rejected);

        N_col(row)       = n;
        K_col(row)       = k;
        Kfrac_col(row)   = k/n;
        Reject_col(row)  = rejectRate;
        MCSE_col(row)    = sqrt(rejectRate*(1-rejectRate)/MC);
        AvgW_col(row)    = mean(wHatRep);
        SdW_col(row)     = std(wHatRep);
        MedianW_col(row) = median(wHatRep);

        replications{row} = struct( ...
            'n', n, ...
            'k', k, ...
            'k_over_n', k/n, ...
            'pvalue', pvalues, ...
            'reject', rejected, ...
            'w_hat', wHatRep);

        fprintf(['  rejection rate = %.4f, MCSE = %.4f, ' ...
                 'mean(w_hat) = %.4f\n'], ...
                 Reject_col(row), MCSE_col(row), AvgW_col(row));
    end
end

toc;

%% Collect and save results

results = table( ...
    N_col, K_col, Kfrac_col, Reject_col, MCSE_col, ...
    AvgW_col, SdW_col, MedianW_col, Perm_col, Level_col, ...
    'VariableNames', { ...
        'n', 'k', 'k_over_n', ...
        'rejection_rate', 'mcse', ...
        'mean_w_hat', 'sd_w_hat', 'median_w_hat', ...
        'permutations', 'nominal_level'});

disp(results);

writetable(results, ...
    'DGP1_beta1_dcov_orderstat_summary.xlsx', ...
    'Sheet', 'Summary');

writetable(results, ...
    'DGP1_beta1_dcov_orderstat_results.csv');

settings = table( ...
    MC, Rperm, level, numWorkers, baseSeed, ...
    'VariableNames', { ...
        'MC_replications', 'permutations', ...
        'nominal_level', 'parallel_workers', 'base_seed'});

writetable(settings, ...
    'DGP1_beta1_dcov_orderstat_summary.xlsx', ...
    'Sheet', 'Settings');

save('DGP1_beta1_dcov_orderstat_replications.mat', ...
    'replications', 'results', 'settings', ...
    'nList', 'kByN', 'MC', 'Rperm', 'level', ...
    'numWorkers', 'baseSeed', '-v7.3');

%% Plot empirical size

figure;
hold on;

for in = 1:numel(nList)
    rows = (results.n == nList(in));

    % Sort by k/n for a clean left-to-right plot.
    [xplot, ord] = sort(results.k_over_n(rows));
    yplot = results.rejection_rate(rows);
    yplot = yplot(ord);

    plot(xplot, yplot, '-o', ...
        'LineWidth', 1.4, ...
        'DisplayName', sprintf('n = %d', nList(in)));
end

yline(level, '--', 'Nominal 5%', 'LineWidth', 1.2);

xlabel('Tail fraction k/n');
ylabel('Rejection probability');
title('DGP 1 (beta=1): size of the global dCov permutation test');
legend('Location','best');
grid on;
hold off;

end


function pval = dcov_permutation_pvalue(X, Z, Rperm)
% DCOV_PERMUTATION_PVALUE
% Monte Carlo permutation p-value for empirical squared distance covariance.
%
% The +1 correction gives
%
%   p = [1 + #{T_perm >= T_obs}] / (Rperm + 1).

X = X(:);
Z = Z(:);

m = numel(X);

if numel(Z) ~= m
    error('X and Z must have the same number of observations.');
end

A = centered_distance_matrix(X);
B = centered_distance_matrix(Z);

Tobs = sum(A .* B, 'all') / (m^2);

nGreaterEqual = 0;

for r = 1:Rperm
    pi = randperm(m);

    % Permuting Z corresponds to applying the same permutation to
    % the rows and columns of its centered distance matrix.
    Bpi = B(pi, pi);

    Tperm = sum(A .* Bpi, 'all') / (m^2);

    nGreaterEqual = nGreaterEqual + (Tperm >= Tobs);
end

pval = (1 + nGreaterEqual) / (Rperm + 1);

end


function A = centered_distance_matrix(V)
% CENTERED_DISTANCE_MATRIX
% Double-centered Euclidean distance matrix.
%
% This gives the usual biased empirical squared distance covariance used
% in the paper.

V = V(:);

D = abs(V - V.');

rowMean = mean(D, 2);
colMean = mean(D, 1);
grandMean = mean(D, 'all');

A = D - rowMean - colMean + grandMean;

end
