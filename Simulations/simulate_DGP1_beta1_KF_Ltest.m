function results = simulate_DGP1_beta1_KF_Ltest()
% SIMULATE_DGP1_BETA1_KF_LTEST
% Monte Carlo size experiment for the Kinsvater-Fried (2017) L-test
% under the SAME DGP 1 used for the dCov simulations.
%
% DGP 1:
%   X ~ Unif[-1,1]
%   s(X) = exp(X/2)
%   Y = s(X)*(U^(-1/2)-1),  U ~ Unif(0,1)
%
% Hence
%   P(Y > y | X=x)
%       = [1 + y/s(x)]^(-2)
%       = s(x)^2 y^(-2){1 - 2*s(x)*y^(-1) + O(y^(-2))}.
%
% The tail exponent is alpha0 = 2, hence the EVI is
%
%       gamma0 = 1/alpha0 = 1/2,
%
% and is constant in x. Thus this is a SIZE experiment.
% The Hall second-order exponent is beta = 1.
%
% -------------------------------------------------------------------------
% Kinsvater-Fried implementation
% -------------------------------------------------------------------------
% The L-test is implemented from Sections 2.1 and 3.1 of
% Kinsvater and Fried (2017):
%
%  1. For a target tail size k, set
%
%         p_kn = (n-k)/(n+1).
%
%  2. Estimate the conditional p_kn quantile threshold u(x) by
%     quantile regression after a monotone transformation.
%
%  3. Retain observations satisfying Y_i > u(X_i), and form
%
%         Z_i = Y_i/u(X_i).
%
%  4. On the relative exceedances, estimate the linear EVI model
%
%         gamma(x) = eta0 + eta1*x
%
%     from 20 regression quantiles of log(Z), equally spaced over
%
%         [1/2, 1-1/40].
%
%     For each probability p,
%
%         eta_hat_p = -beta_hat_p/log(1-p).
%
%     The estimates are combined using the optimal L-weights
%
%         w_opt = (1'A_p^{-1}1)^{-1} A_p^{-1}1.
%
%  5. Test H0: eta1 = 0 with the Kinsvater-Fried L-statistic.
%
% -------------------------------------------------------------------------
% IMPORTANT DGP-1 simplification
% -------------------------------------------------------------------------
% For THIS DGP,
%
%     log(Y) = X/2 + log(U^(-1/2)-1),
%
% exactly. Therefore the Box-Cox transformation lambda = 0 (log) gives an
% EXACT linear conditional-quantile model at every probability level.
%
% The baseline code therefore uses lambda = 0 as the known population-
% correct transformation. This is an ORACLE/FAVORABLE implementation for
% Kinsvater-Fried: it removes transformation-estimation noise and focuses
% the comparison on the tail-homogeneity test itself.
%
% If desired later, the Mu-He/Kinsvater-Fried Box-Cox lambda estimator can
% be added as a separate robustness exercise. For the main DGP-1 comparison,
% using the correct log transformation is conservative in the sense that it
% gives the competing procedure an advantage.
%
% -------------------------------------------------------------------------
% Tail sample sizes
% -------------------------------------------------------------------------
% Same k-values as in the revised dCov simulation:
%
%       n = 2000: k = 100, 50
%       n = 5000: k = 200, 100
%
% Note that Kinsvater-Fried use a covariate-dependent threshold, so the
% realized number m of relative exceedances is typically close to, but not
% exactly equal to, k. Their paper notes that in simulations it is usually
% between k-2 and k+2. This code uses the ACTUAL realized m in the
% finite-sample studentization and reports its mean and standard deviation.
%
% -------------------------------------------------------------------------
% Quantile regression
% -------------------------------------------------------------------------
% To keep the file self-contained and avoid relying on a third-party
% quantile-regression package, this code exploits the scalar-regressor
% structure of DGP 1. For a fixed slope b1, the minimizing intercept is the
% empirical tau-quantile of y-b1*x. The resulting one-dimensional convex
% profile objective is minimized with fminbnd.
%
% OUTPUT FILES:
%   DGP1_beta1_KF_Ltest_summary.xlsx
%   DGP1_beta1_KF_Ltest_results.csv
%   DGP1_beta1_KF_Ltest_replications.mat

%% User settings

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
kByN  = [100,  50; ...
         200, 100];

MC    = 1000;
level = 0.05;

% Kinsvater-Fried recommendation:
% 20 probabilities equally spaced over [1/2, 1-1/40].
ell   = 20;
pGrid = linspace(0.5, 1 - 1/40, ell).';

% Precompute A_p and optimal L-weights.
A = kf_A_matrix(pGrid);
oneVec = ones(ell,1);

if rcond(A) > 1e-12
    Ainv1 = A \ oneVec;
else
    Ainv1 = pinv(A) * oneVec;
end

wOpt = Ainv1 / (oneVec' * Ainv1);
aWeight = wOpt' * A * wOpt;

fprintf('DGP 1 (beta=1): Kinsvater-Fried (2017) L-test\n');
fprintf('Null EVI gamma(x) = 0.5 for all x\n');
fprintf('Using population-correct Box-Cox transformation lambda = 0 (log)\n');
fprintf('L-estimator probabilities: %d points on [0.5, 0.975]\n', ell);
for in = 1:numel(nList)
    fprintf('  n = %d: target k = %d, %d\n', ...
        nList(in), kByN(in,1), kByN(in,2));
end
fprintf('\n');

%% Storage

nDesigns = numel(nList) * size(kByN,2);

N_col          = zeros(nDesigns,1);
K_col          = zeros(nDesigns,1);
Reject_col     = zeros(nDesigns,1);
MCSE_col       = zeros(nDesigns,1);
MeanM_col      = zeros(nDesigns,1);
SdM_col        = zeros(nDesigns,1);
MeanTL_col     = zeros(nDesigns,1);
SdTL_col       = zeros(nDesigns,1);
MeanEta0_col   = zeros(nDesigns,1);
MeanEta1_col   = zeros(nDesigns,1);
Fail_col       = zeros(nDesigns,1);
Level_col      = level * ones(nDesigns,1);

replications = cell(nDesigns,1);

%% Monte Carlo

row = 0;
tic;

for in = 1:numel(nList)

    n = nList(in);

    for ik = 1:size(kByN,2)

        k = kByN(in,ik);
        designID = (in-1)*size(kByN,2) + ik;

        rejected = false(MC,1);
        pvalues  = NaN(MC,1);
        TLrep    = NaN(MC,1);
        mRep     = NaN(MC,1);
        eta0Rep  = NaN(MC,1);
        eta1Rep  = NaN(MC,1);
        failed   = false(MC,1);

        fprintf('Running n = %d, target k = %d ...\n', n, k);

        parfor rep = 1:MC

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
            % Kinsvater-Fried covariate-dependent threshold
            % -------------------------------------------------------------
            % p_kn = (n-k)/(n+1)
            pkn = (n - k) / (n + 1);

            % For DGP 1, lambda = 0 is exactly correct:
            % g_lambda(Y) = log(Y).
            logY = log(Y);

            % Quantile regression:
            % Q_{log Y}(pkn | X=x) = beta0 + beta1*x.
            betaThresh = qr_scalar_profile(logY, X, pkn);

            logUhat = betaThresh(1) + betaThresh(2)*X;
            uhat = exp(logUhat);

            idx = (Y > uhat);

            Xe = X(idx);
            Ze = Y(idx) ./ uhat(idx);

            % Ze should exceed one up to numerical precision.
            good = isfinite(Xe) & isfinite(Ze) & (Ze > 1);
            Xe = Xe(good);
            Ze = Ze(good);

            m = numel(Ze);
            mRep(rep) = m;

            % Need enough observations for 20 QR fits and a stable
            % two-parameter covariance estimate.
            if m < 30
                failed(rep) = true;
                continue;
            end

            % -------------------------------------------------------------
            % Kinsvater-Fried L-estimator
            % -------------------------------------------------------------
            logZ = log(Ze);
            etaByP = NaN(2, ell);

            thisFailed = false;

            for jp = 1:ell
                pp = pGrid(jp);

                betaP = qr_scalar_profile(logZ, Xe, pp);

                etaP = -betaP / log(1 - pp);

                if any(~isfinite(etaP))
                    thisFailed = true;
                    break;
                end

                etaByP(:,jp) = etaP;
            end

            if thisFailed
                failed(rep) = true;
                continue;
            end

            etaL = etaByP * wOpt;

            eta0Rep(rep) = etaL(1);
            eta1Rep(rep) = etaL(2);

            % -------------------------------------------------------------
            % Plug-in asymptotic covariance
            %
            % J = m^{-1} sum x_j x_j'
            % H = m^{-1} sum x_j x_j' / gamma_hat(x_j)
            %
            % Sigma_eta = (w'A_p w) H^{-1} J H^{-1}
            % -------------------------------------------------------------
            Xmat = [ones(m,1), Xe];
            gammaHat = Xmat * etaL;

            % The Pareto EVI must be positive over the selected design.
            if any(~isfinite(gammaHat)) || any(gammaHat <= 1e-8)
                failed(rep) = true;
                continue;
            end

            Jhat = (Xmat' * Xmat) / m;

            Hhat = zeros(2,2);
            for j = 1:m
                xx = Xmat(j,:).';
                Hhat = Hhat + (xx * xx.') / gammaHat(j);
            end
            Hhat = Hhat / m;

            if rcond(Hhat) < 1e-10
                failed(rep) = true;
                continue;
            end

            HinvJHinv = Hhat \ Jhat / Hhat;
            SigmaEta = aWeight * HinvJHinv;

            slopeVarScale = SigmaEta(2,2);

            if ~isfinite(slopeVarScale) || slopeVarScale <= 0
                failed(rep) = true;
                continue;
            end

            % The paper's T^L = sqrt(k)*eta_hat_1/sigma_hat_1.
            % Because the estimated conditional threshold can select m
            % observations rather than exactly target k, we use the actual
            % effective tail count m here.
            TL = sqrt(m) * etaL(2) / sqrt(slopeVarScale);

            % Two-sided normal p-value without requiring normcdf().
            pval = erfc(abs(TL) / sqrt(2));

            TLrep(rep) = TL;
            pvalues(rep) = pval;
            rejected(rep) = (pval <= level);
        end

        used = ~failed & isfinite(pvalues);
        nUsed = sum(used);

        row = row + 1;

        N_col(row)      = n;
        K_col(row)      = k;
        Fail_col(row)   = MC - nUsed;

        if nUsed > 0
            rejectRate = mean(rejected(used));

            Reject_col(row)   = rejectRate;
            MCSE_col(row)     = sqrt(rejectRate*(1-rejectRate)/nUsed);
            MeanM_col(row)    = mean(mRep(used));
            SdM_col(row)      = std(mRep(used));
            MeanTL_col(row)   = mean(TLrep(used));
            SdTL_col(row)     = std(TLrep(used));
            MeanEta0_col(row) = mean(eta0Rep(used));
            MeanEta1_col(row) = mean(eta1Rep(used));
        else
            Reject_col(row)   = NaN;
            MCSE_col(row)     = NaN;
            MeanM_col(row)    = NaN;
            SdM_col(row)      = NaN;
            MeanTL_col(row)   = NaN;
            SdTL_col(row)     = NaN;
            MeanEta0_col(row) = NaN;
            MeanEta1_col(row) = NaN;
        end

        replications{row} = struct( ...
            'n', n, ...
            'target_k', k, ...
            'pvalue', pvalues, ...
            'reject', rejected, ...
            'T_L', TLrep, ...
            'effective_tail_n', mRep, ...
            'eta0_hat', eta0Rep, ...
            'eta1_hat', eta1Rep, ...
            'failed', failed);

        fprintf(['  rejection rate = %.4f, MCSE = %.4f, ' ...
                 'mean effective m = %.2f, failures = %d/%d\n'], ...
                 Reject_col(row), MCSE_col(row), ...
                 MeanM_col(row), Fail_col(row), MC);
    end
end

toc;

%% Collect and save

results = table( ...
    N_col, K_col, Reject_col, MCSE_col, ...
    MeanM_col, SdM_col, MeanTL_col, SdTL_col, ...
    MeanEta0_col, MeanEta1_col, Fail_col, Level_col, ...
    'VariableNames', { ...
        'n', 'target_k', 'rejection_rate', 'mcse', ...
        'mean_effective_tail_n', 'sd_effective_tail_n', ...
        'mean_TL', 'sd_TL', ...
        'mean_eta0_hat', 'mean_eta1_hat', ...
        'failures', 'nominal_level'});

disp(results);

writetable(results, ...
    'DGP1_beta1_KF_Ltest_summary.xlsx', ...
    'Sheet', 'Summary');

writetable(results, ...
    'DGP1_beta1_KF_Ltest_results.csv');

settings = table( ...
    MC, level, numWorkers, baseSeed, ell, ...
    'VariableNames', { ...
        'MC_replications', 'nominal_level', ...
        'parallel_workers', 'base_seed', ...
        'L_probabilities'});

writetable(settings, ...
    'DGP1_beta1_KF_Ltest_summary.xlsx', ...
    'Sheet', 'Settings');

save('DGP1_beta1_KF_Ltest_replications.mat', ...
    'replications', 'results', 'settings', ...
    'nList', 'kByN', 'MC', 'level', ...
    'pGrid', 'wOpt', 'A', 'aWeight', ...
    'numWorkers', 'baseSeed', '-v7.3');

%% Plot size

figure;
hold on;

for in = 1:numel(nList)
    rows = (results.n == nList(in));

    [kPlot, ord] = sort(results.target_k(rows));
    rPlot = results.rejection_rate(rows);
    rPlot = rPlot(ord);

    plot(kPlot, rPlot, '-o', ...
        'LineWidth', 1.4, ...
        'DisplayName', sprintf('n = %d', nList(in)));
end

yline(level, '--', 'Nominal 5%', 'LineWidth', 1.2);

xlabel('Target tail sample size k');
ylabel('Rejection probability');
title('DGP 1 (beta=1): Kinsvater-Fried L-test size');
legend('Location','best');
grid on;
hold off;

end


function A = kf_A_matrix(pGrid)
% Kinsvater-Fried (2017), Proposition 1:
%
% a(p_i,p_j) =
%   [min(p_i,p_j)-p_i p_j] /
%   [(1-p_i)(1-p_j) log(1-p_i) log(1-p_j)].

pGrid = pGrid(:);
ell = numel(pGrid);

A = zeros(ell,ell);

for i = 1:ell
    for j = 1:ell
        pi_ = pGrid(i);
        pj_ = pGrid(j);

        num = min(pi_, pj_) - pi_*pj_;
        den = (1-pi_)*(1-pj_)*log(1-pi_)*log(1-pj_);

        A(i,j) = num / den;
    end
end

% Numerical symmetrization.
A = 0.5*(A + A.');

end


function beta = qr_scalar_profile(y, x, tau)
% QR_SCALAR_PROFILE
% Quantile regression with an intercept and ONE scalar regressor:
%
%     Q_y(tau | x) = beta0 + beta1*x.
%
% For a fixed slope b1, an optimal intercept is an empirical tau-quantile
% of y-b1*x. Hence the two-dimensional QR problem can be profiled to a
% one-dimensional convex minimization over b1.
%
% This avoids dependence on a third-party quantile-regression package.

y = y(:);
x = x(:);

n = numel(y);

if numel(x) ~= n
    error('x and y must have the same length.');
end

if ~(tau > 0 && tau < 1)
    error('tau must lie strictly between 0 and 1.');
end

% OLS slope as a convenient center for the search interval.
xc = x - mean(x);
yc = y - mean(y);

den = sum(xc.^2);

if den > 1e-14
    bOLS = sum(xc .* yc) / den;
else
    bOLS = 0;
end

sx = std(x);
sy = std(y);

slopeScale = sy / max(sx, 1e-8);
radius = 10 * max([1, abs(bOLS), slopeScale]);

lb = bOLS - radius;
ub = bOLS + radius;

opts = optimset( ...
    'Display', 'off', ...
    'TolX', 1e-7, ...
    'MaxIter', 150, ...
    'MaxFunEvals', 300);

obj = @(b1) qr_profile_loss(b1, y, x, tau);

% Expand the bracket if the optimum is suspiciously close to an endpoint.
for attempt = 1:3

    [b1, ~] = fminbnd(obj, lb, ub, opts);

    span = ub - lb;
    nearLeft  = (b1 - lb) < 0.01*span;
    nearRight = (ub - b1) < 0.01*span;

    if ~(nearLeft || nearRight)
        break;
    end

    radius = 2*radius;
    lb = bOLS - radius;
    ub = bOLS + radius;
end

r = y - b1*x;
b0 = empirical_tau_quantile(r, tau);

beta = [b0; b1];

end


function loss = qr_profile_loss(b1, y, x, tau)

r = y - b1*x;
b0 = empirical_tau_quantile(r, tau);

u = r - b0;

loss = sum(u .* (tau - (u < 0)));

end


function q = empirical_tau_quantile(v, tau)
% An empirical tau-quantile suitable for the quantile-regression profile
% objective. Any value in the empirical quantile interval minimizes the
% check loss; we use the left order-statistic convention.

v = sort(v(:));
n = numel(v);

idx = ceil(tau*n);
idx = max(1, min(n, idx));

q = v(idx);

end
