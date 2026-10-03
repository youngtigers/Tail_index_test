function results = simulate_DGP2_KF_Ltest()
% SIMULATE_DGP2_KF_LTEST
% DGP 2: scalar local alternatives, Kinsvater-Fried (2017) L-test.
%
% X ~ Unif[-1,1]
% kappa(x) = exp(x/2)
% g(x) = sqrt(3)*x
% alpha_{n,c}(x) = 2 + c*g(x)/sqrt(k)
% Y = {kappa(X)/U}^{1/alpha_{n,c}(X)}.
%
% Since gamma(x)=1/alpha(x),
%
%   gamma(x)
%     = 1/2 - c*sqrt(3)*x/(4*sqrt(k)) + O(k^{-1}),
%
% so DGP 2 is deliberately close to the linear-EVI alternative targeted
% by the Kinsvater-Fried L-test.
%
% We use the same implementation as for DGP 1:
%   - target tail size k
%   - p_kn=(n-k)/(n+1)
%   - log transformation (lambda=0)
%   - covariate-dependent QR threshold
%   - 20 regression-quantile probabilities on [0.5,0.975]
%   - optimal L-weights
%   - two-sided normal test of the EVI slope
%
% Under c=0, lambda=0 gives an exact linear transformed-quantile model.
% Under c>0, the departure is local and the model is locally linear.
%
% Current development settings:
%   MC = 1000
%
% Outputs:
%   DGP2_KF_Ltest_summary.xlsx
%   DGP2_KF_Ltest_results.csv
%   DGP2_KF_Ltest_replications.mat

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
level = 0.05;

ell   = 20;
pGrid = linspace(0.5, 1 - 1/40, ell).';

A = kf_A_matrix(pGrid);
oneVec = ones(ell,1);
if rcond(A) > 1e-12
    Ainv1 = A \ oneVec;
else
    Ainv1 = pinv(A) * oneVec;
end
wOpt = Ainv1 / (oneVec' * Ainv1);
aWeight = wOpt' * A * wOpt;

fprintf('DGP 2: Kinsvater-Fried L-test\n');
fprintf('alpha_{n,c}(x) = 2 + c*sqrt(3)*x/sqrt(k)\n');
fprintf('MC = %d\n\n', MC);

%% Storage
nDesigns = numel(nList)*size(kByN,2)*numel(cList);

N_col        = zeros(nDesigns,1);
K_col        = zeros(nDesigns,1);
C_col        = zeros(nDesigns,1);
Reject_col   = zeros(nDesigns,1);
MCSE_col     = zeros(nDesigns,1);
MeanM_col    = zeros(nDesigns,1);
SdM_col      = zeros(nDesigns,1);
MeanTL_col   = zeros(nDesigns,1);
SdTL_col     = zeros(nDesigns,1);
MeanEta0_col = zeros(nDesigns,1);
MeanEta1_col = zeros(nDesigns,1);
Fail_col     = zeros(nDesigns,1);
Level_col    = level*ones(nDesigns,1);

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
            TLrep    = NaN(MC,1);
            mRep     = NaN(MC,1);
            eta0Rep  = NaN(MC,1);
            eta1Rep  = NaN(MC,1);
            failed   = false(MC,1);

            fprintf('Running n=%d, k=%d, c=%.1f ...\n',n,k,c);

            parfor rep = 1:MC
                rng(baseSeed + 100000*designID + rep,'twister');

                % ----- Generate DGP 2 -----
                X = 2*rand(n,1)-1;
                U = rand(n,1);
                U(U==0) = realmin;

                kappa = exp(X/2);
                alpha = 2 + (c/sqrt(k))*sqrt(3)*X;

                if any(alpha <= 0)
                    error('DGP 2 generated a nonpositive tail exponent.');
                end

                Y = (kappa ./ U).^(1 ./ alpha);
                logY = log(Y);

                % ----- KF covariate-dependent threshold -----
                pkn = (n-k)/(n+1);

                betaThresh = qr_scalar_profile(logY, X, pkn);
                logUhat = betaThresh(1) + betaThresh(2)*X;
                uhat = exp(logUhat);

                idx = (Y > uhat);
                Xe = X(idx);
                Ze = Y(idx)./uhat(idx);

                good = isfinite(Xe) & isfinite(Ze) & (Ze > 1);
                Xe = Xe(good);
                Ze = Ze(good);

                m = numel(Ze);
                mRep(rep) = m;

                if m < 30
                    failed(rep) = true;
                    continue;
                end

                % ----- KF L-estimator -----
                logZ = log(Ze);
                etaByP = NaN(2,ell);
                thisFailed = false;

                for jp = 1:ell
                    pp = pGrid(jp);
                    betaP = qr_scalar_profile(logZ, Xe, pp);
                    etaP = -betaP/log(1-pp);

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

                etaL = etaByP*wOpt;
                eta0Rep(rep) = etaL(1);
                eta1Rep(rep) = etaL(2);

                Xmat = [ones(m,1), Xe];
                gammaHat = Xmat*etaL;

                if any(~isfinite(gammaHat)) || any(gammaHat <= 1e-8)
                    failed(rep) = true;
                    continue;
                end

                Jhat = (Xmat'*Xmat)/m;

                Hhat = zeros(2,2);
                for j = 1:m
                    xx = Xmat(j,:).';
                    Hhat = Hhat + (xx*xx.')/gammaHat(j);
                end
                Hhat = Hhat/m;

                if rcond(Hhat) < 1e-10
                    failed(rep) = true;
                    continue;
                end

                HinvJHinv = Hhat \ Jhat / Hhat;
                SigmaEta = aWeight*HinvJHinv;
                slopeVarScale = SigmaEta(2,2);

                if ~isfinite(slopeVarScale) || slopeVarScale <= 0
                    failed(rep) = true;
                    continue;
                end

                TL = sqrt(m)*etaL(2)/sqrt(slopeVarScale);
                pval = erfc(abs(TL)/sqrt(2));

                TLrep(rep) = TL;
                pvalues(rep) = pval;
                rejected(rep) = (pval <= level);
            end

            used = ~failed & isfinite(pvalues);
            nUsed = sum(used);

            N_col(row) = n;
            K_col(row) = k;
            C_col(row) = c;
            Fail_col(row) = MC-nUsed;

            if nUsed > 0
                rr = mean(rejected(used));
                Reject_col(row)   = rr;
                MCSE_col(row)     = sqrt(rr*(1-rr)/nUsed);
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
                'n',n,'k',k,'c',c, ...
                'pvalue',pvalues,'reject',rejected,'T_L',TLrep, ...
                'effective_tail_n',mRep,'eta0_hat',eta0Rep, ...
                'eta1_hat',eta1Rep,'failed',failed);

            fprintf('  rejection rate = %.4f, mean m = %.2f, failures=%d\n', ...
                Reject_col(row),MeanM_col(row),Fail_col(row));
        end
    end
end

toc;

%% Save
results = table( ...
    N_col,K_col,C_col,Reject_col,MCSE_col,MeanM_col,SdM_col, ...
    MeanTL_col,SdTL_col,MeanEta0_col,MeanEta1_col,Fail_col,Level_col, ...
    'VariableNames', { ...
    'n','k','c','rejection_rate','mcse','mean_effective_tail_n', ...
    'sd_effective_tail_n','mean_TL','sd_TL','mean_eta0_hat', ...
    'mean_eta1_hat','failures','nominal_level'});

disp(results);

writetable(results,'DGP2_KF_Ltest_summary.xlsx','Sheet','Summary');
writetable(results,'DGP2_KF_Ltest_results.csv');

settings = table(MC,level,numWorkers,baseSeed,ell, ...
    'VariableNames', {'MC_replications','nominal_level', ...
    'parallel_workers','base_seed','L_probabilities'});
writetable(settings,'DGP2_KF_Ltest_summary.xlsx','Sheet','Settings');

save('DGP2_KF_Ltest_replications.mat', ...
    'replications','results','settings','nList','kByN','cList', ...
    'MC','level','pGrid','wOpt','A','aWeight', ...
    'numWorkers','baseSeed','-v7.3');

%% Plot
figure;
hold on;
for in = 1:numel(nList)
    for ik = 1:size(kByN,2)
        n = nList(in);
        k = kByN(in,ik);
        rows = (results.n==n) & (results.k==k);
        plot(results.c(rows),results.rejection_rate(rows),'-o', ...
            'LineWidth',1.3,'DisplayName',sprintf('n=%d, k=%d',n,k));
    end
end
yline(level,'--','Nominal 5%','LineWidth',1.1);
xlabel('Local-alternative parameter c');
ylabel('Rejection probability');
title('DGP 2: Kinsvater-Fried L-test power');
legend('Location','best');
grid on;
hold off;

end


function A = kf_A_matrix(pGrid)

pGrid = pGrid(:);
ell = numel(pGrid);
A = zeros(ell,ell);

for i=1:ell
    for j=1:ell
        pi_ = pGrid(i);
        pj_ = pGrid(j);
        num = min(pi_,pj_) - pi_*pj_;
        den = (1-pi_)*(1-pj_)*log(1-pi_)*log(1-pj_);
        A(i,j) = num/den;
    end
end

A = 0.5*(A+A.');

end


function beta = qr_scalar_profile(y,x,tau)

y = y(:);
x = x(:);

xc = x-mean(x);
yc = y-mean(y);
den = sum(xc.^2);

if den > 1e-14
    bOLS = sum(xc.*yc)/den;
else
    bOLS = 0;
end

sx = std(x);
sy = std(y);
slopeScale = sy/max(sx,1e-8);
radius = 10*max([1,abs(bOLS),slopeScale]);

lb = bOLS-radius;
ub = bOLS+radius;

opts = optimset('Display','off','TolX',1e-7, ...
    'MaxIter',150,'MaxFunEvals',300);

obj = @(b1) qr_profile_loss(b1,y,x,tau);

for attempt=1:3
    [b1,~] = fminbnd(obj,lb,ub,opts);
    span = ub-lb;
    if (b1-lb)>=0.01*span && (ub-b1)>=0.01*span
        break;
    end
    radius = 2*radius;
    lb = bOLS-radius;
    ub = bOLS+radius;
end

r = y-b1*x;
b0 = empirical_tau_quantile(r,tau);
beta = [b0;b1];

end


function loss = qr_profile_loss(b1,y,x,tau)

r = y-b1*x;
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
