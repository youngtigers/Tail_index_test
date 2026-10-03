function results = simulate_DGP2_local_J23()
% SIMULATE_DGP2_LOCAL_J23
% DGP 2: proposed dCov test with LOCALLY BALANCED TAIL SELECTION.
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
% LOCAL-WINDOW PROCEDURE:
%   1. Run the procedure for J = 2 and J = 3 empirical-quantile windows.
%   2. Sort X and divide the observations into J consecutive blocks whose
%      sizes differ by at most one.
%   3. Allocate the total tail count k across the J windows as evenly as
%      possible, again differing by at most one.  The allocation is chosen
%      symmetrically when J=3, e.g.
%          k=100 -> (33,34,33),
%          k= 50 -> (17,16,17),
%          k=200 -> (67,66,67).
%      Thus the pooled sample always contains exactly k observations.
%   4. In window j, retain its k_j largest Y observations and let w_j be
%      the next local order statistic:
%          w_j = Y_{(n_j-k_j):n_j}.
%      For each retained observation form
%          Z_i^* = log(Y_i / w_j).
%   5. Pool all retained observations across windows.
%   6. Compute dCov between pooled X and pooled Z^*, and calibrate by
%      GLOBAL permutation of the pooled Z^* values.
%
% Settings:
%   JList = [2,3]
%   MC    = 1000
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
%   DGP2_dcov_local_J23_summary.xlsx
%   DGP2_dcov_local_J23_results.csv
%   DGP2_dcov_local_J23_replications.mat

%% User settings
numWorkers = 8;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', numWorkers);
elseif pool.NumWorkers ~= numWorkers
    delete(pool);
    parpool('local', numWorkers);
end

baseSeed = 20260916;

nList = [2000, 5000];
kByN  = [100,  50; ...
         200, 100];
cList = [0, 1, 2, 4];
JList = [2, 3];

MC    = 1000;
Rperm = 299;
level = 0.05;

fprintf('DGP 2: proposed dCov test with locally balanced X windows\n');
fprintf('J = 2 and 3\n');
fprintf('alpha_{n,c}(x) = 2 + c*sqrt(3)*x/sqrt(k)\n');
fprintf('MC = %d, permutations = %d\n\n', MC, Rperm);

%% Storage
nDesigns = numel(JList) * numel(nList) * size(kByN,2) * numel(cList);

N_col          = zeros(nDesigns,1);
K_col          = zeros(nDesigns,1);
J_col          = zeros(nDesigns,1);
C_col          = zeros(nDesigns,1);
Reject_col     = zeros(nDesigns,1);
MCSE_col       = zeros(nDesigns,1);
AvgW_col       = zeros(nDesigns,1);
SdAvgW_col     = zeros(nDesigns,1);
MinW_col       = zeros(nDesigns,1);
MaxW_col       = zeros(nDesigns,1);
AlphaMin_col   = zeros(nDesigns,1);
AlphaMax_col   = zeros(nDesigns,1);
MinNWin_col    = zeros(nDesigns,1);
MaxNWin_col    = zeros(nDesigns,1);
MinKWin_col    = zeros(nDesigns,1);
MaxKWin_col    = zeros(nDesigns,1);
Perm_col       = Rperm * ones(nDesigns,1);
Level_col      = level * ones(nDesigns,1);

replications = cell(nDesigns,1);

row = 0;
tic;

for iJ = 1:numel(JList)
    J = JList(iJ);

    for in = 1:numel(nList)
        n = nList(in);
        nPerWindow = balanced_counts_symmetric(n, J);

        for ik = 1:size(kByN,2)
            k = kByN(in,ik);
            kPerWindow = balanced_counts_symmetric(k, J);

            if any(kPerWindow >= nPerWindow)
                error('Need k_j < n_j in every local window.');
            end

            for ic = 1:numel(cList)
                c = cList(ic);

                row = row + 1;
                designID = row;

                rejected      = false(MC,1);
                pvalues       = NaN(MC,1);
                avgWHatRep    = NaN(MC,1);
                minWHatRep    = NaN(MC,1);
                maxWHatRep    = NaN(MC,1);
                localWHatRep  = NaN(MC,J);

                fprintf(['Running J=%d, n=%d, k=%d, c=%.1f, ' ...
                         'n/window=[%s], k/window=[%s] ...\n'], ...
                    J, n, k, c, num2str(nPerWindow), num2str(kPerWindow));

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

                    % -----------------------------------------------------
                    % Divide X into J empirical-quantile windows.
                    % Window sizes are as equal as possible.
                    % -----------------------------------------------------
                    [~, xOrder] = sort(X, 'ascend');

                    Xe = NaN(k,1);
                    Ze = NaN(k,1);
                    wLocal = NaN(J,1);

                    posData = 0;
                    posTail = 0;

                    for j = 1:J
                        nj = nPerWindow(j);
                        kj = kPerWindow(j);

                        winIdx = xOrder((posData+1):(posData+nj));
                        posData = posData + nj;

                        Xj = X(winIdx);
                        Yj = Y(winIdx);

                        % Retain top k_j Y's in this X-window and use
                        % the next local order statistic as w_j.
                        [YjDesc, yOrderLocal] = sort(Yj, 'descend');

                        wj = YjDesc(kj + 1);
                        localTop = yOrderLocal(1:kj);

                        sel = (posTail+1):(posTail+kj);

                        Xe(sel) = Xj(localTop);
                        Ze(sel) = log(Yj(localTop) ./ wj);

                        wLocal(j) = wj;
                        posTail = posTail + kj;
                    end

                    if posData ~= n
                        error('Local windows did not use exactly n observations.');
                    end
                    if posTail ~= k
                        error('Local selection did not produce exactly k observations.');
                    end

                    % ----- GLOBAL dCov permutation test on pooled sample -----
                    pval = dcov_permutation_pvalue(Xe, Ze, Rperm);

                    pvalues(rep)  = pval;
                    rejected(rep) = (pval <= level);

                    localWHatRep(rep,:) = wLocal.';
                    avgWHatRep(rep) = mean(wLocal);
                    minWHatRep(rep) = min(wLocal);
                    maxWHatRep(rep) = max(wLocal);
                end

                rejectRate = mean(rejected);

                N_col(row)          = n;
                K_col(row)          = k;
                J_col(row)          = J;
                C_col(row)          = c;
                Reject_col(row)     = rejectRate;
                MCSE_col(row)       = sqrt(rejectRate*(1-rejectRate)/MC);
                AvgW_col(row)       = mean(avgWHatRep);
                SdAvgW_col(row)     = std(avgWHatRep);
                MinW_col(row)       = mean(minWHatRep);
                MaxW_col(row)       = mean(maxWHatRep);
                AlphaMin_col(row)   = 2 - c*sqrt(3)/sqrt(k);
                AlphaMax_col(row)   = 2 + c*sqrt(3)/sqrt(k);
                MinNWin_col(row)    = min(nPerWindow);
                MaxNWin_col(row)    = max(nPerWindow);
                MinKWin_col(row)    = min(kPerWindow);
                MaxKWin_col(row)    = max(kPerWindow);

                replications{row} = struct( ...
                    'n', n, ...
                    'k', k, ...
                    'J', J, ...
                    'n_per_window', nPerWindow, ...
                    'k_per_window', kPerWindow, ...
                    'c', c, ...
                    'pvalue', pvalues, ...
                    'reject', rejected, ...
                    'local_w_hat', localWHatRep, ...
                    'average_local_w_hat', avgWHatRep, ...
                    'minimum_local_w_hat', minWHatRep, ...
                    'maximum_local_w_hat', maxWHatRep);

                fprintf('  rejection rate = %.4f, MCSE = %.4f\n', ...
                    Reject_col(row), MCSE_col(row));
            end
        end
    end
end

toc;

%% Save
results = table( ...
    N_col, K_col, J_col, C_col, ...
    AlphaMin_col, AlphaMax_col, ...
    MinNWin_col, MaxNWin_col, MinKWin_col, MaxKWin_col, ...
    Reject_col, MCSE_col, ...
    AvgW_col, SdAvgW_col, MinW_col, MaxW_col, ...
    Perm_col, Level_col, ...
    'VariableNames', { ...
        'n','k','windows','c', ...
        'alpha_min','alpha_max', ...
        'min_n_per_window','max_n_per_window', ...
        'min_k_per_window','max_k_per_window', ...
        'rejection_rate','mcse', ...
        'mean_average_local_w_hat','sd_average_local_w_hat', ...
        'mean_min_local_w_hat','mean_max_local_w_hat', ...
        'permutations','nominal_level'});

disp(results);

writetable(results, ...
    'DGP2_dcov_local_J23_summary.xlsx', 'Sheet', 'Summary');
writetable(results, ...
    'DGP2_dcov_local_J23_results.csv');

settings = table(MC, Rperm, level, numWorkers, baseSeed, ...
    'VariableNames', { ...
        'MC_replications','permutations','nominal_level', ...
        'parallel_workers','base_seed'});
writetable(settings, ...
    'DGP2_dcov_local_J23_summary.xlsx', 'Sheet', 'Settings');

Jsettings = table(JList(:), ...
    'VariableNames', {'windows'});
writetable(Jsettings, ...
    'DGP2_dcov_local_J23_summary.xlsx', 'Sheet', 'J_values');

save('DGP2_dcov_local_J23_replications.mat', ...
    'replications','results','settings', ...
    'nList','kByN','cList','JList', ...
    'MC','Rperm','level','numWorkers','baseSeed','-v7.3');

%% Plot power: separate figure for each J
for iJ = 1:numel(JList)
    J = JList(iJ);

    figure;
    hold on;

    for in = 1:numel(nList)
        for ik = 1:size(kByN,2)
            n = nList(in);
            k = kByN(in,ik);

            rows = (results.windows == J) & ...
                   (results.n == n) & (results.k == k);

            plot(results.c(rows), results.rejection_rate(rows), '-o', ...
                'LineWidth',1.3, ...
                'DisplayName',sprintf('n=%d, k=%d',n,k));
        end
    end

    yline(level,'--','Nominal 5%','LineWidth',1.1);
    xlabel('Local-alternative parameter c');
    ylabel('Rejection probability');
    title(sprintf('DGP 2: dCov power with J=%d local X windows',J));
    legend('Location','best');
    grid on;
    hold off;
end

end


function counts = balanced_counts_symmetric(total, J)
% Split TOTAL into J integer counts that differ by at most one.
% For J=3, place remainders symmetrically when possible:
%   remainder 1 -> middle window,
%   remainder 2 -> two outside windows.

base = floor(total/J);
r = mod(total,J);
counts = base * ones(1,J);

if r == 0
    return;
end

if J == 2
    % Any one-unit imbalance is unavoidable when total is odd.
    counts(1:r) = counts(1:r) + 1;
elseif J == 3
    if r == 1
        counts(2) = counts(2) + 1;
    elseif r == 2
        counts([1,3]) = counts([1,3]) + 1;
    end
else
    % Generic fallback, not used in this program.
    counts(1:r) = counts(1:r) + 1;
end

end


function pval = dcov_permutation_pvalue(X, Z, Rperm)
% Global permutation test after pooling all local-window exceedances.

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
    Bpi = B(pi,pi);

    Tperm = sum(A .* Bpi, 'all') / (m^2);

    nGreaterEqual = nGreaterEqual + (Tperm >= Tobs);
end

pval = (1 + nGreaterEqual)/(Rperm + 1);

end


function A = centered_distance_matrix(V)

V = V(:);

D = abs(V - V.');

rowMean   = mean(D,2);
colMean   = mean(D,1);
grandMean = mean(D,'all');

A = D - rowMean - colMean + grandMean;

end
