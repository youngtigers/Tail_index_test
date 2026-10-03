function results = application_WangLi2013(dataFile)
% APPLICATION_WANGLI2013
% French MTPL application: Wang and Li (2013) EVI-constancy test.
%
% Covariates:
%   X = (BonusMalus, VehPower)
%
% This deliberately uses the SAME Wang-Li implementation as in the
% simulation section of the paper:
%
%   - lambda = 0, i.e. log(Y) transformation
%   - eta = 0.1
%   - upper conditional quantile regressions of log(Y) on
%         (1, BonusMalus, VehPower)
%   - monotone rearrangement across fitted quantile levels
%   - conditional Hill-type EVI estimator gamma_hat(x)
%   - pooled EVI gamma_P
%   - T_n = n^{-1} sum_i {gamma_hat(X_i)-gamma_P}^2
%   - Corollary 3.1 chi-square calibration:
%
%         k*T_n/gamma_P^2  ~  chi-square_2
%
% We use two tail sizes:
%
%   k_1 = round(0.05*n)    (upper 5%)
%   k_2 = round(0.025*n)   (upper 2.5%)
%
% These correspond approximately to threshold quantiles 0.95 and 0.975.
%
% IMPORTANT:
% This is computationally intensive. For the larger k, the code fits
% roughly k-floor(n^eta)+1 upper quantile regressions.
%
% Outputs:
%   application_WangLi2013_results.xlsx
%   application_WangLi2013_results.csv
%   application_WangLi2013_details.mat

if nargin < 1 || isempty(dataFile)
    dataFile = 'freMTPL2_application.csv';
end

%% Settings
level = 0.05;
etaTrunc = 0.1;
doRearrangement = true;

% Two tail fractions proposed for the application.
kFracList = [0.05, 0.025];

%% Read data
dat = readtable(dataFile);

requiredVars = {'ClaimAmount','BonusMalus','VehPower'};

for j = 1:numel(requiredVars)
    if ~ismember(requiredVars{j},dat.Properties.VariableNames)
        error('Variable %s is missing from the data.',requiredVars{j});
    end
end

Y = dat.ClaimAmount;
Xraw = [dat.BonusMalus,dat.VehPower];

valid = ...
    isfinite(Y) & ...
    Y > 0 & ...
    all(isfinite(Xraw),2);

Y = Y(valid);
Xraw = Xraw(valid,:);

n = numel(Y);

%% Standardize / center covariates
% Corollary 3.1 is simplest after centering the nonconstant regressors.
Xmean = mean(Xraw,1);
Xsd = std(Xraw,0,1);

if any(Xsd <= 0)
    error('At least one covariate has zero variance.');
end

X = (Xraw-Xmean)./Xsd;
Xmat = [ones(n,1),X];

logY = log(Y);

%% Tail sizes
kList = round(n*kFracList);

% Avoid accidental duplicates for small samples.
kList = unique(kList,'stable');

fprintf('\n============================================================\n');
fprintf('French MTPL application: Wang-Li (2013)\n');
fprintf('============================================================\n');
fprintf('Usable positive claims: n = %d\n',n);
fprintf('Covariates: BonusMalus, VehPower\n');
fprintf('lambda = 0; eta = %.2f; rearrangement = %d\n', ...
    etaTrunc,doRearrangement);
fprintf('Tail sizes: ');
fprintf('%d ',kList);
fprintf('\n\n');

%% Storage
nK = numel(kList);

K_col = zeros(nK,1);
Kfrac_col = zeros(nK,1);
EquivalentQ_col = zeros(nK,1);

J0_col = zeros(nK,1);
NumQR_col = zeros(nK,1);

GammaP_col = NaN(nK,1);
Tn_col = NaN(nK,1);
Stat_col = NaN(nK,1);
Pvalue_col = NaN(nK,1);

GammaMin_col = NaN(nK,1);
GammaMax_col = NaN(nK,1);
GammaSd_col = NaN(nK,1);

details = cell(nK,1);

%% Main loop
for ik = 1:nK

    k = kList(ik);

    j0 = floor(n^etaTrunc);
    j0 = max(j0,1);

    if k <= j0
        error('Need k > floor(n^eta).');
    end

    % j counts upper order positions. This gives increasing tau.
    jAsc = (k:-1:j0).';
    tauAsc = (n-jAsc)/(n+1);

    nQR = numel(tauAsc);

    fprintf('------------------------------------------------------------\n');
    fprintf('k = %d  (k/n = %.5f; equivalent q = %.5f)\n', ...
        k,k/n,1-k/n);
    fprintf('j0 = %d; number of QR fits = %d\n',j0,nQR);
    fprintf('------------------------------------------------------------\n');

    %% Upper-tail quantile regressions of log(Y)
    betaMat = NaN(3,nQR);

    betaOLS = Xmat\logY;
    slopeStart = betaOLS(2:end);

    tic;

    for jt = 1:nQR

        tau = tauAsc(jt);

        beta = qr_multivariate_profile( ...
            logY,X,tau,slopeStart);

        if any(~isfinite(beta))
            error('Quantile regression failed at k=%d, tau=%.8f.', ...
                k,tau);
        end

        betaMat(:,jt) = beta;

        % Warm start for the next, slightly higher, quantile.
        slopeStart = beta(2:end);

        if mod(jt,100)==0 || jt==nQR
            fprintf('  QR %d / %d completed (tau = %.6f)\n', ...
                jt,nQR,tau);
        end
    end

    qrTime = toc;

    fprintf('QR stage completed in %.1f minutes.\n',qrTime/60);

    %% Predicted conditional log-quantiles
    % n x nQR can be a large matrix, but it is needed for row-wise
    % monotone rearrangement, matching the simulation implementation.
    logQhat = Xmat*betaMat;

    if doRearrangement
        logQhat = sort(logQhat,2,'ascend');
    end

    %% Conditional Hill-type EVI estimator
    logQbase = logQhat(:,1);

    gammaHat = ...
        sum(logQhat-logQbase,2)/(k-j0);

    gammaP = mean(gammaHat);

    if ~isfinite(gammaP) || gammaP <= 1e-10 || ...
            any(~isfinite(gammaHat))
        error('Invalid Wang-Li EVI estimates for k=%d.',k);
    end

    %% EVI constancy test
    Tn = mean((gammaHat-gammaP).^2);

    % p = 3 regressors including intercept, so p-1 = 2.
    stat = k*Tn/(gammaP^2);

    % Upper-tail probability of chi-square_2.
    pval = gammainc(stat/2,1,'upper');

    fprintf('\n');
    fprintf('gamma_P         = %.8f\n',gammaP);
    fprintf('sd(gamma_hat)   = %.8f\n',std(gammaHat));
    fprintf('min gamma_hat   = %.8f\n',min(gammaHat));
    fprintf('max gamma_hat   = %.8f\n',max(gammaHat));
    fprintf('T_n             = %.10f\n',Tn);
    fprintf('chi-square stat = %.8f\n',stat);
    fprintf('p-value         = %.6f\n\n',pval);

    %% Store
    K_col(ik) = k;
    Kfrac_col(ik) = k/n;
    EquivalentQ_col(ik) = 1-k/n;

    J0_col(ik) = j0;
    NumQR_col(ik) = nQR;

    GammaP_col(ik) = gammaP;
    Tn_col(ik) = Tn;
    Stat_col(ik) = stat;
    Pvalue_col(ik) = pval;

    GammaMin_col(ik) = min(gammaHat);
    GammaMax_col(ik) = max(gammaHat);
    GammaSd_col(ik) = std(gammaHat);

    details{ik} = struct( ...
        'k',k, ...
        'k_over_n',k/n, ...
        'equivalent_q',1-k/n, ...
        'j0',j0, ...
        'tau_grid',tauAsc, ...
        'beta_quantile_regression',betaMat, ...
        'gamma_hat',gammaHat, ...
        'gamma_pooled',gammaP, ...
        'Tn',Tn, ...
        'chi2_stat',stat, ...
        'pvalue',pval, ...
        'qr_minutes',qrTime/60);
end

%% Results table
results = table( ...
    K_col,Kfrac_col,EquivalentQ_col, ...
    J0_col,NumQR_col, ...
    GammaP_col,GammaSd_col,GammaMin_col,GammaMax_col, ...
    Tn_col,Stat_col,Pvalue_col, ...
    'VariableNames',{ ...
    'k','k_over_n','equivalent_threshold_quantile', ...
    'j0_floor_n_eta','number_QR_fits', ...
    'gamma_pooled','sd_gamma_hat','min_gamma_hat','max_gamma_hat', ...
    'Tn','chi2_stat','pvalue'});

fprintf('\n============================================================\n');
fprintf('FINAL WANG-LI RESULTS\n');
fprintf('============================================================\n\n');

disp(results);

%% Save
writetable(results, ...
    'application_WangLi2013_results.xlsx', ...
    'Sheet','Results');

writetable(results, ...
    'application_WangLi2013_results.csv');

settings = table( ...
    n,etaTrunc,level,doRearrangement, ...
    'VariableNames',{ ...
    'n','eta','nominal_level','rearrangement'});

writetable(settings, ...
    'application_WangLi2013_results.xlsx', ...
    'Sheet','Settings');

save('application_WangLi2013_details.mat', ...
    'results','details','settings','Xmean','Xsd','kList', ...
    'kFracList','etaTrunc','level','doRearrangement','-v7.3');

end


%% ============================================================
% Local functions
% ============================================================

function beta = qr_multivariate_profile(y,X,tau,slopeStart)

y = y(:);
[n,d] = size(X);

if nargin < 4 || numel(slopeStart) ~= d || ...
        any(~isfinite(slopeStart))

    Z = [ones(n,1),X];
    bOLS = Z\y;
    slopeStart = bOLS(2:end);
end

slopeStart = slopeStart(:);

opts = optimset( ...
    'Display','off', ...
    'TolX',2e-6, ...
    'TolFun',1e-6, ...
    'MaxIter',180, ...
    'MaxFunEvals',450);

obj = @(b) qr_profile_loss_multi(b,y,X,tau);

[bSlope,~,exitflag] = ...
    fminsearch(obj,slopeStart,opts);

if exitflag == 0

    opts2 = optimset( ...
        opts, ...
        'MaxIter',300, ...
        'MaxFunEvals',750);

    [bSlope,~,~] = ...
        fminsearch(obj,bSlope,opts2);
end

r = y-X*bSlope;

b0 = empirical_tau_quantile(r,tau);

beta = [b0;bSlope(:)];

end


function loss = qr_profile_loss_multi(b,y,X,tau)

r = y-X*b(:);

b0 = empirical_tau_quantile(r,tau);

u = r-b0;

loss = sum(u.*(tau-(u<0)));

end


function q = empirical_tau_quantile(v,tau)

v = sort(v(:));

n = numel(v);

idx = ceil(tau*n);

idx = max(1,min(n,idx));

q = v(idx);

end
