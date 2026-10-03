function results = application_KF_Ltest(dataFile,kFracList,level)
% APPLICATION_KF_LTEST
% French MTPL application: Kinsvater-Fried (2017) L-test.
%
% Covariates:
%   X = (BonusMalus, VehPower)
%
% Tail sizes:
%   k_1 = round(0.05*n)
%   k_2 = round(0.025*n)
% by default.
%
% Implementation matches the simulation competitor:
%   - lambda = 0 (log transformation)
%   - conditional threshold QR of log(Y) on (1,X)
%   - 20 exceedance QR probabilities on [0.5, 0.975]
%   - optimal L-weights
%   - joint Wald test H0: both EVI slopes are zero
%
% Outputs:
%   application_KF_Ltest_results.xlsx
%   application_KF_Ltest_results.csv

if nargin < 1 || isempty(dataFile)
    dataFile = 'freMTPL2_application.csv';
end
if nargin < 2 || isempty(kFracList)
    kFracList = [0.05,0.025];
end
if nargin < 3 || isempty(level)
    level = 0.05;
end

%% Read and clean data
dat = readtable(dataFile);

requiredVars = {'ClaimAmount','BonusMalus','VehPower'};
for j=1:numel(requiredVars)
    if ~ismember(requiredVars{j},dat.Properties.VariableNames)
        error('Variable %s is missing from the data.',requiredVars{j});
    end
end

Y = dat.ClaimAmount;
Xraw = [dat.BonusMalus,dat.VehPower];

valid = isfinite(Y) & Y>0 & all(isfinite(Xraw),2);
Y = Y(valid);
Xraw = Xraw(valid,:);

n = numel(Y);
kList = round(n*kFracList);

% Standardize for numerical stability.
Xmean = mean(Xraw,1);
Xsd = std(Xraw,0,1);
X = (Xraw-Xmean)./Xsd;

logY = log(Y);

%% KF L-weights
ell = 20;
pGrid = linspace(0.5,1-1/40,ell).';

A = kf_A_matrix(pGrid);
oneVec = ones(ell,1);

if rcond(A)>1e-12
    Ainv1 = A\oneVec;
else
    Ainv1 = pinv(A)*oneVec;
end

wOpt = Ainv1/(oneVec'*Ainv1);
aWeight = wOpt'*A*wOpt;

fprintf('\nFrench MTPL application: Kinsvater-Fried L-test\n');
fprintf('Usable positive claims: n=%d\n',n);
fprintf('X=(BonusMalus, VehPower)\n');
fprintf('Tail sizes: ');
fprintf('%d ',kList);
fprintf('\n\n');

%% Storage
nK = numel(kList);

K_col = zeros(nK,1);
Kfrac_col = zeros(nK,1);
EquivalentQ_col = zeros(nK,1);
EffectiveTail_col = zeros(nK,1);
Wald_col = NaN(nK,1);
P_col = NaN(nK,1);
Eta0_col = NaN(nK,1);
EtaBonus_col = NaN(nK,1);
EtaVehPower_col = NaN(nK,1);

for ik=1:nK
    k = kList(ik);
    qeq = 1-k/n;

    fprintf('============================================\n');
    fprintf('k=%d (k/n=%.5f; equivalent q=%.5f)\n',k,k/n,qeq);
    fprintf('============================================\n');

    %% Conditional threshold QR
    pkn = (n-k)/(n+1);

    Z0 = [ones(n,1),X];
    betaOLS = Z0\logY;
    betaThresh = qr_multivariate_profile(logY,X,pkn,betaOLS(2:end));

    logUhat = [ones(n,1),X]*betaThresh;
    uhat = exp(logUhat);

    idx = (Y>uhat);
    Xe = X(idx,:);
    Ze = Y(idx)./uhat(idx);

    good = all(isfinite(Xe),2) & isfinite(Ze) & Ze>1;
    Xe = Xe(good,:);
    Ze = Ze(good);

    m = numel(Ze);

    if m<35
        warning('Too few effective tail observations at k=%d.',k);
        K_col(ik)=k;
        Kfrac_col(ik)=k/n;
        EquivalentQ_col(ik)=qeq;
        EffectiveTail_col(ik)=m;
        continue;
    end

    %% L-estimator
    logZ = log(Ze);
    etaByP = NaN(3,ell);

    Zols = [ones(m,1),Xe];
    betaStart = Zols\logZ;
    slopeStart = betaStart(2:end);

    failed = false;

    for jp=1:ell
        pp = pGrid(jp);
        betaP = qr_multivariate_profile(logZ,Xe,pp,slopeStart);

        if any(~isfinite(betaP))
            failed = true;
            break;
        end

        etaP = -betaP/log(1-pp);
        etaByP(:,jp) = etaP;
        slopeStart = betaP(2:end);
    end

    if failed
        warning('QR failure at k=%d.',k);
        K_col(ik)=k;
        Kfrac_col(ik)=k/n;
        EquivalentQ_col(ik)=qeq;
        EffectiveTail_col(ik)=m;
        continue;
    end

    etaL = etaByP*wOpt;

    Xmat = [ones(m,1),Xe];
    gammaHat = Xmat*etaL;

    if any(~isfinite(gammaHat)) || any(gammaHat<=1e-8)
        warning('Invalid fitted EVI values at k=%d.',k);
        K_col(ik)=k;
        Kfrac_col(ik)=k/n;
        EquivalentQ_col(ik)=qeq;
        EffectiveTail_col(ik)=m;
        continue;
    end

    Jhat = (Xmat'*Xmat)/m;

    Hhat = zeros(3,3);
    for j=1:m
        xx = Xmat(j,:).';
        Hhat = Hhat+(xx*xx.')/gammaHat(j);
    end
    Hhat = Hhat/m;

    if rcond(Hhat)<1e-10
        warning('Nearly singular H matrix at k=%d.',k);
        K_col(ik)=k;
        Kfrac_col(ik)=k/n;
        EquivalentQ_col(ik)=qeq;
        EffectiveTail_col(ik)=m;
        continue;
    end

    HinvJHinv = Hhat\Jhat/Hhat;
    SigmaEta = aWeight*HinvJHinv;

    b = etaL(2:3);
    V = SigmaEta(2:3,2:3);

    if rcond(V)<1e-10 || any(~isfinite(V),'all')
        warning('Nearly singular Wald covariance at k=%d.',k);
        K_col(ik)=k;
        Kfrac_col(ik)=k/n;
        EquivalentQ_col(ik)=qeq;
        EffectiveTail_col(ik)=m;
        continue;
    end

    W = m*(b'*(V\b));
    pval = gammainc(W/2,1,'upper');

    fprintf('effective tail n=%d\n',m);
    fprintf('eta slopes: BonusMalus=%.6f, VehPower=%.6f\n',etaL(2),etaL(3));
    fprintf('KF Wald=%.6f, p=%.4f\n\n',W,pval);

    K_col(ik)=k;
    Kfrac_col(ik)=k/n;
    EquivalentQ_col(ik)=qeq;
    EffectiveTail_col(ik)=m;
    Wald_col(ik)=W;
    P_col(ik)=pval;
    Eta0_col(ik)=etaL(1);
    EtaBonus_col(ik)=etaL(2);
    EtaVehPower_col(ik)=etaL(3);
end

results = table( ...
    K_col,Kfrac_col,EquivalentQ_col,EffectiveTail_col,Wald_col,P_col, ...
    Eta0_col,EtaBonus_col,EtaVehPower_col, ...
    'VariableNames',{ ...
    'k','k_over_n','equivalent_threshold_quantile','effective_tail_n', ...
    'KF_wald','KF_pvalue','eta0_hat','eta_bonusmalus_hat', ...
    'eta_vehpower_hat'});

disp(results);

writetable(results,'application_KF_Ltest_results.xlsx','Sheet','Results');
writetable(results,'application_KF_Ltest_results.csv');

end


function A = kf_A_matrix(pGrid)

pGrid = pGrid(:);
ell = numel(pGrid);
A = zeros(ell,ell);

for i=1:ell
    for j=1:ell
        pi_ = pGrid(i);
        pj_ = pGrid(j);
        num = min(pi_,pj_)-pi_*pj_;
        den = (1-pi_)*(1-pj_)*log(1-pi_)*log(1-pj_);
        A(i,j) = num/den;
    end
end

A = 0.5*(A+A.');
end


function beta = qr_multivariate_profile(y,X,tau,slopeStart)

y = y(:);
[n,d] = size(X);

if nargin<4 || numel(slopeStart)~=d || any(~isfinite(slopeStart))
    Z = [ones(n,1),X];
    bOLS = Z\y;
    slopeStart = bOLS(2:end);
end

slopeStart = slopeStart(:);

opts = optimset('Display','off','TolX',2e-6,'TolFun',1e-6, ...
    'MaxIter',180,'MaxFunEvals',450);

obj = @(b) qr_profile_loss_multi(b,y,X,tau);
[bSlope,~,exitflag] = fminsearch(obj,slopeStart,opts);

if exitflag==0
    opts2 = optimset(opts,'MaxIter',300,'MaxFunEvals',750);
    [bSlope,~,~] = fminsearch(obj,bSlope,opts2);
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
