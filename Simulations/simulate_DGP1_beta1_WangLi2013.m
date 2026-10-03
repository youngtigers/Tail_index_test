function results = simulate_DGP1_beta1_WangLi2013()
% SIMULATE_DGP1_BETA1_WANGLI2013
% Monte Carlo size experiment for the Wang-Li (2013) test of a constant
% extreme-value index, under the SAME DGP 1 used for the dCov simulations.
%
% DGP 1:
%   X ~ Unif[-1,1]
%   s(X) = exp(X/2)
%   Y = s(X)*(U^(-1/2)-1),  U ~ Unif(0,1)
%
% Hence
%   P(Y > y | X=x)
%       = [1 + y/s(x)]^(-2)
%       = s(x)^2*y^(-2)*{1 - 2*s(x)*y^(-1) + O(y^(-2))}.
%
% The tail exponent is alpha0 = 2, so the conditional EVI is
%
%       gamma(x) = 1/alpha0 = 1/2,
%
% for every x. Thus this is a SIZE experiment.
% The Hall second-order exponent is beta = 1.
%
% -------------------------------------------------------------------------
% Wang and Li (2013) implementation
% -------------------------------------------------------------------------
% Their three-stage method:
%
%   Stage 1: estimate a Box-Cox transformation parameter lambda.
%   Stage 2: estimate intermediate conditional quantiles by linear QR on
%            the transformed scale, transform them back, and rearrange
%            estimated quantile curves if needed.
%   Stage 3: estimate gamma(x) using a Hill estimator applied to the
%            pseudo upper-order statistics formed from those estimated
%            conditional quantiles:
%
%       gamma_hat(x)
%         = 1/(k-floor(n^eta))
%           sum_{j=floor(n^eta)}^k
%             log{ Q_hat_Y(tau_{n-j}|x) /
%                  Q_hat_Y(tau_{n-k}|x) }.
%
% They test constancy using
%
%       T_n = n^{-1} sum_i {gamma_hat(X_i)-gamma_hat_P}^2,
%
% where gamma_hat_P = n^{-1} sum_i gamma_hat(X_i).
%
% -------------------------------------------------------------------------
% IMPORTANT DGP-1 SIMPLIFICATION / FAVORABLE IMPLEMENTATION
% -------------------------------------------------------------------------
% For THIS DGP,
%
%     log(Y) = X/2 + log(U^(-1/2)-1),
%
% exactly. Therefore lambda = 0 (log transformation) is the population-
% correct Wang-Li transformation, and the transformed conditional
% quantiles are EXACTLY linear in X.
%
% The code therefore uses lambda = 0 rather than estimating lambda in
% every Monte Carlo replication. This is an ORACLE/FAVORABLE version of
% Wang-Li for DGP 1: it removes transformation-estimation noise and makes
% the comparison conservative with respect to our dCov procedure.
%
% In addition, for lambda = 0 the transformed EVI gamma_0^* equals zero.
% Wang and Li's Corollary 3.1 then gives the simple null calibration
%
%       k*T_n / gamma_hat_P^2  ==>  chi-square_{p-1},
%
% after centering the non-intercept covariates. Here p=2 (intercept plus
% one scalar covariate), so the reference distribution is chi-square_1.
% Therefore the more cumbersome heterogeneous critical-value simulation
% from their Theorem 3.3 is NOT needed for this particular DGP.
%
% -------------------------------------------------------------------------
% Tail tuning values
% -------------------------------------------------------------------------
% Same k-values as in the revised dCov and Kinsvater-Fried simulations:
%
%       n = 2000: k = 100, 50
%       n = 5000: k = 200, 100
%
% Wang-Li use eta = 0.1 in their empirical work; we do the same.
%
% -------------------------------------------------------------------------
% Computational note
% -------------------------------------------------------------------------
% The Wang-Li estimator requires many upper-tail quantile regressions:
% roughly k-floor(n^eta)+1 per replication. Consequently this file is
% substantially more computationally intensive than the dCov or
% Kinsvater-Fried files.
%
% To keep the file self-contained, scalar quantile regression is solved
% by profiling out the intercept. For a fixed slope b1, the minimizing
% intercept is an empirical tau-quantile of y-b1*x, leaving a
% one-dimensional convex minimization in b1.
%
% OUTPUT FILES:
%   DGP1_beta1_WangLi2013_summary.xlsx
%   DGP1_beta1_WangLi2013_results.csv
%   DGP1_beta1_WangLi2013_replications.mat

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

MC       = 1000;
level    = 0.05;
etaTrunc = 0.1;     % Wang-Li empirical choice
doRearrangement = true;

fprintf('DGP 1 (beta=1): Wang and Li (2013) constant-EVI test\n');
fprintf('True gamma(x) = 0.5 for all x\n');
fprintf('Using population-correct Box-Cox lambda = 0 (log transform)\n');
fprintf('Using Corollary 3.1 chi-square calibration (df = 1)\n');
fprintf('eta = %.2f\n', etaTrunc);
for in = 1:numel(nList)
    fprintf('  n = %d: k = %d, %d\n', ...
        nList(in), kByN(in,1), kByN(in,2));
end
fprintf('\n');

%% Storage

nDesigns = numel(nList) * size(kByN,2);

N_col            = zeros(nDesigns,1);
K_col            = zeros(nDesigns,1);
J0_col           = zeros(nDesigns,1);
NumQR_col        = zeros(nDesigns,1);
Reject_col       = zeros(nDesigns,1);
MCSE_col         = zeros(nDesigns,1);
MeanGammaP_col   = zeros(nDesigns,1);
SdGammaP_col     = zeros(nDesigns,1);
MeanTn_col       = zeros(nDesigns,1);
MeanStat_col     = zeros(nDesigns,1);
SdStat_col       = zeros(nDesigns,1);
Fail_col         = zeros(nDesigns,1);
Level_col        = level * ones(nDesigns,1);

replications = cell(nDesigns,1);

%% Monte Carlo experiment

row = 0;
tic;

for in = 1:numel(nList)

    n = nList(in);

    for ik = 1:size(kByN,2)

        k = kByN(in,ik);
        designID = (in-1)*size(kByN,2) + ik;

        % Wang-Li truncation index [n^eta].
        j0 = floor(n^etaTrunc);

        if j0 < 1
            j0 = 1;
        end

        if k <= j0
            error('Need k > floor(n^eta).');
        end

        % Equation (2.8) requires tau_{n-j}, j=j0,...,k.
        %
        % For rearrangement it is convenient to order the tau levels from
        % low to high. Since tau_{n-j}=(n-j)/(n+1), this means using
        % j = k, k-1, ..., j0.
        jAsc = (k:-1:j0).';
        tauAsc = (n - jAsc) / (n + 1);
        nQR = numel(tauAsc);

        rejected = false(MC,1);
        pvalues  = NaN(MC,1);
        gammaPRep = NaN(MC,1);
        TnRep     = NaN(MC,1);
        statRep   = NaN(MC,1);
        failed    = false(MC,1);

        fprintf('Running n = %d, k = %d (%d QR fits/rep) ...\n', ...
            n, k, nQR);

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

            logY = log(Y);

            % Center X, as required for the simple Corollary 3.1
            % calibration. With an intercept, centering is only a
            % reparameterization and leaves fitted conditional quantiles
            % unchanged.
            Xc = X - mean(X);
            Xmat = [ones(n,1), Xc];

            % -------------------------------------------------------------
            % Stage 2: upper-tail quantile regressions on log(Y)
            %
            % Since lambda = 0, Lambda_lambda(Y)=log(Y).
            % We only fit the upper quantiles needed in equation (2.8).
            % -------------------------------------------------------------
            betaMat = NaN(2, nQR);

            % OLS slope provides a common starting center.
            xc = Xc;
            yc = logY;
            denOLS = sum((xc-mean(xc)).^2);
            if denOLS > 1e-14
                bStart = sum((xc-mean(xc)).*(yc-mean(yc))) / denOLS;
            else
                bStart = 0;
            end

            thisFailed = false;

            for jt = 1:nQR

                tau = tauAsc(jt);

                beta = qr_scalar_profile_warm(logY, Xc, tau, bStart);

                if any(~isfinite(beta))
                    thisFailed = true;
                    break;
                end

                betaMat(:,jt) = beta;
                bStart = beta(2);   % warm start for neighboring quantile
            end

            if thisFailed
                failed(rep) = true;
                continue;
            end

            % Predicted log conditional quantiles at all observed X_i.
            %
            % For lambda=0:
            %   Q_hat_Y(tau|x) = exp(x'beta_hat(tau)),
            % so log Q_hat_Y = x'beta_hat.
            logQhat = Xmat * betaMat;

            % Wang-Li recommend rearrangement if estimated conditional
            % quantile curves cross. Because exp(.) is monotone, sorting
            % logQhat is equivalent to sorting Qhat itself.
            if doRearrangement
                logQhat = sort(logQhat, 2, 'ascend');
            end

            % -------------------------------------------------------------
            % Stage 3: Wang-Li conditional Hill estimator, equation (2.8)
            %
            % tauAsc(1) = tau_{n-k}, the denominator quantile.
            %
            % The equation has denominator k-floor(n^eta), even though
            % the sum contains k-floor(n^eta)+1 terms; the denominator
            % term itself contributes zero.
            % -------------------------------------------------------------
            logQbase = logQhat(:,1);

            gammaHat = sum(logQhat - logQbase, 2) / (k - j0);

            gammaP = mean(gammaHat);

            if ~isfinite(gammaP) || gammaP <= 1e-10 || ...
                    any(~isfinite(gammaHat))
                failed(rep) = true;
                continue;
            end

            % -------------------------------------------------------------
            % Wang-Li test statistic, Section 3.4
            % -------------------------------------------------------------
            Tn = mean((gammaHat - gammaP).^2);

            % Corollary 3.1:
            %   k*Tn / gamma^2 -> chi-square_{p-1}
            %
            % Here p=2, hence df=1.
            df = 1;
            stat = k * Tn / (gammaP^2);

            % Upper-tail chi-square p-value:
            % P(ChiSq_df >= stat)
            pval = gammainc(stat/2, df/2, 'upper');

            gammaPRep(rep) = gammaP;
            TnRep(rep) = Tn;
            statRep(rep) = stat;
            pvalues(rep) = pval;
            rejected(rep) = (pval <= level);
        end

        used = ~failed & isfinite(pvalues);
        nUsed = sum(used);

        row = row + 1;

        N_col(row)     = n;
        K_col(row)     = k;
        J0_col(row)    = j0;
        NumQR_col(row) = nQR;
        Fail_col(row)  = MC - nUsed;

        if nUsed > 0
            rejectRate = mean(rejected(used));

            Reject_col(row)     = rejectRate;
            MCSE_col(row)       = sqrt(rejectRate*(1-rejectRate)/nUsed);
            MeanGammaP_col(row) = mean(gammaPRep(used));
            SdGammaP_col(row)   = std(gammaPRep(used));
            MeanTn_col(row)     = mean(TnRep(used));
            MeanStat_col(row)   = mean(statRep(used));
            SdStat_col(row)     = std(statRep(used));
        else
            Reject_col(row)     = NaN;
            MCSE_col(row)       = NaN;
            MeanGammaP_col(row) = NaN;
            SdGammaP_col(row)   = NaN;
            MeanTn_col(row)     = NaN;
            MeanStat_col(row)   = NaN;
            SdStat_col(row)     = NaN;
        end

        replications{row} = struct( ...
            'n', n, ...
            'k', k, ...
            'j0', j0, ...
            'tau_grid', tauAsc, ...
            'pvalue', pvalues, ...
            'reject', rejected, ...
            'gamma_pooled', gammaPRep, ...
            'Tn', TnRep, ...
            'chi2_stat', statRep, ...
            'failed', failed);

        fprintf(['  rejection rate = %.4f, MCSE = %.4f, ' ...
                 'mean gamma_P = %.4f, failures = %d/%d\n'], ...
                 Reject_col(row), MCSE_col(row), ...
                 MeanGammaP_col(row), Fail_col(row), MC);
    end
end

toc;

%% Collect and save results

results = table( ...
    N_col, K_col, J0_col, NumQR_col, ...
    Reject_col, MCSE_col, ...
    MeanGammaP_col, SdGammaP_col, MeanTn_col, ...
    MeanStat_col, SdStat_col, Fail_col, Level_col, ...
    'VariableNames', { ...
        'n', 'k', 'j0_floor_n_eta', 'quantile_regressions_per_rep', ...
        'rejection_rate', 'mcse', ...
        'mean_gamma_pooled', 'sd_gamma_pooled', 'mean_Tn', ...
        'mean_chi2_stat', 'sd_chi2_stat', ...
        'failures', 'nominal_level'});

disp(results);

writetable(results, ...
    'DGP1_beta1_WangLi2013_summary.xlsx', ...
    'Sheet', 'Summary');

writetable(results, ...
    'DGP1_beta1_WangLi2013_results.csv');

settings = table( ...
    MC, level, etaTrunc, numWorkers, baseSeed, doRearrangement, ...
    'VariableNames', { ...
        'MC_replications', 'nominal_level', 'eta', ...
        'parallel_workers', 'base_seed', 'rearrangement'});

writetable(settings, ...
    'DGP1_beta1_WangLi2013_summary.xlsx', ...
    'Sheet', 'Settings');

save('DGP1_beta1_WangLi2013_replications.mat', ...
    'replications', 'results', 'settings', ...
    'nList', 'kByN', 'MC', 'level', 'etaTrunc', ...
    'numWorkers', 'baseSeed', 'doRearrangement', '-v7.3');

%% Plot empirical size

figure;
hold on;

for in = 1:numel(nList)

    rows = (results.n == nList(in));

    [kPlot, ord] = sort(results.k(rows));
    rPlot = results.rejection_rate(rows);
    rPlot = rPlot(ord);

    plot(kPlot, rPlot, '-o', ...
        'LineWidth', 1.4, ...
        'DisplayName', sprintf('n = %d', nList(in)));
end

yline(level, '--', 'Nominal 5%', 'LineWidth', 1.2);

xlabel('Tail tuning parameter k');
ylabel('Rejection probability');
title('DGP 1 (beta=1): Wang-Li (2013) constant-EVI test');
legend('Location','best');
grid on;
hold off;

end


function beta = qr_scalar_profile_warm(y, x, tau, bCenter)
% QR_SCALAR_PROFILE_WARM
% Quantile regression with intercept and ONE scalar regressor:
%
%     Q_y(tau | x) = beta0 + beta1*x.
%
% For fixed beta1, the minimizing intercept is an empirical tau-quantile
% of y-beta1*x. This profiles the QR problem to one dimension.
%
% bCenter is a warm-start center, typically the slope from a neighboring
% quantile regression.

y = y(:);
x = x(:);

n = numel(y);

if numel(x) ~= n
    error('x and y must have the same length.');
end

if ~(tau > 0 && tau < 1)
    error('tau must lie strictly between 0 and 1.');
end

if ~isfinite(bCenter)
    bCenter = 0;
end

sx = std(x);
sy = std(y);

slopeScale = sy / max(sx, 1e-8);

% Start with a moderate bracket around the neighboring-quantile slope.
radius = 2.5 * max([0.25, slopeScale, abs(bCenter)/2]);

opts = optimset( ...
    'Display', 'off', ...
    'TolX', 5e-7, ...
    'MaxIter', 120, ...
    'MaxFunEvals', 240);

obj = @(b1) qr_profile_loss(b1, y, x, tau);

for attempt = 1:4

    lb = bCenter - radius;
    ub = bCenter + radius;

    [b1, ~] = fminbnd(obj, lb, ub, opts);

    span = ub - lb;
    nearLeft  = (b1 - lb) < 0.005*span;
    nearRight = (ub - b1) < 0.005*span;

    if ~(nearLeft || nearRight)
        break;
    end

    radius = 2*radius;
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
% EMPIRICAL_TAU_QUANTILE
% Left empirical tau-quantile. Any point in the empirical quantile
% interval minimizes the check loss for the profiled intercept.

v = sort(v(:));
n = numel(v);

idx = ceil(tau*n);
idx = max(1, min(n, idx));

q = v(idx);

end
