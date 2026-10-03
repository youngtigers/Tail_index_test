function results = simulate_DGP2_WangLi2013()
% SIMULATE_DGP2_WANGLI2013
% DGP 2: scalar local alternatives, Wang and Li (2013) constant-EVI test.
%
% X ~ Unif[-1,1]
% kappa(x) = exp(x/2)
% g(x) = sqrt(3)*x
% alpha_{n,c}(x) = 2 + c*g(x)/sqrt(k)
% Y = {kappa(X)/U}^{1/alpha_{n,c}(X)}.
%
% Hence c=0 is the constant-EVI null, while c>0 gives a local departure.
%
% The implementation reuses the DGP-1 Wang-Li code:
%   - lambda = 0 (log transformation)
%   - eta = 0.1
%   - upper conditional QR estimates
%   - rearrangement of estimated conditional quantile curves
%   - conditional Hill estimator gamma_hat(x)
%   - T_n = n^{-1} sum_i {gamma_hat(X_i)-gamma_hat_P}^2
%   - Corollary 3.1 calibration:
%
%         k*T_n/gamma_hat_P^2  ~  chi-square_1
%
% Under c=0, log(Y) is exactly linear-quantile in X. Under c>0 the
% alternative is local; gamma(x)=1/alpha(x) is locally linear:
%
%   gamma(x)=1/2-c*sqrt(3)*x/(4*sqrt(k))+O(k^{-1}).
%
% Current development setting:
%   MC = 1000.
%
% Outputs:
%   DGP2_WangLi2013_summary.xlsx
%   DGP2_WangLi2013_results.csv
%   DGP2_WangLi2013_replications.mat

%% User settings
numWorkers = 48;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local',numWorkers);
elseif pool.NumWorkers ~= numWorkers
    delete(pool);
    parpool('local',numWorkers);
end

baseSeed = 20260914;

nList = [2000,5000];
kByN  = [100, 50; ...
         200,100];
cList = [0,1,2,4];

MC       = 1000;
level    = 0.05;
etaTrunc = 0.1;
doRearrangement = true;

fprintf('DGP 2: Wang and Li (2013) constant-EVI test\n');
fprintf('alpha_{n,c}(x)=2+c*sqrt(3)*x/sqrt(k)\n');
fprintf('MC=%d, eta=%.2f\n\n',MC,etaTrunc);

%% Storage
nDesigns = numel(nList)*size(kByN,2)*numel(cList);

N_col          = zeros(nDesigns,1);
K_col          = zeros(nDesigns,1);
C_col          = zeros(nDesigns,1);
J0_col         = zeros(nDesigns,1);
NumQR_col      = zeros(nDesigns,1);
Reject_col     = zeros(nDesigns,1);
MCSE_col       = zeros(nDesigns,1);
MeanGammaP_col = zeros(nDesigns,1);
SdGammaP_col   = zeros(nDesigns,1);
MeanTn_col     = zeros(nDesigns,1);
MeanStat_col   = zeros(nDesigns,1);
SdStat_col     = zeros(nDesigns,1);
Fail_col       = zeros(nDesigns,1);
Level_col      = level*ones(nDesigns,1);

replications = cell(nDesigns,1);

row = 0;
tic;

for in=1:numel(nList)
    n = nList(in);

    for ik=1:size(kByN,2)
        k = kByN(in,ik);

        j0 = floor(n^etaTrunc);
        j0 = max(j0,1);

        if k <= j0
            error('Need k > floor(n^eta).');
        end

        jAsc = (k:-1:j0).';
        tauAsc = (n-jAsc)/(n+1);
        nQR = numel(tauAsc);

        for ic=1:numel(cList)
            c = cList(ic);

            row = row+1;
            designID = row;

            rejected  = false(MC,1);
            pvalues   = NaN(MC,1);
            gammaPRep = NaN(MC,1);
            TnRep     = NaN(MC,1);
            statRep   = NaN(MC,1);
            failed    = false(MC,1);

            fprintf('Running n=%d, k=%d, c=%.1f (%d QR fits/rep) ...\n', ...
                n,k,c,nQR);

            parfor rep=1:MC
                rng(baseSeed+100000*designID+rep,'twister');

                % ----- Generate DGP 2 -----
                X = 2*rand(n,1)-1;
                U = rand(n,1);
                U(U==0)=realmin;

                kappa = exp(X/2);
                alpha = 2 + (c/sqrt(k))*sqrt(3)*X;

                if any(alpha<=0)
                    error('DGP 2 generated a nonpositive tail exponent.');
                end

                Y = (kappa./U).^(1./alpha);
                logY = log(Y);

                % Center X for Corollary 3.1.
                Xc = X-mean(X);
                Xmat = [ones(n,1),Xc];

                % ----- Stage 2: upper-tail QR on log(Y) -----
                betaMat = NaN(2,nQR);

                denOLS = sum((Xc-mean(Xc)).^2);
                if denOLS > 1e-14
                    bStart = sum((Xc-mean(Xc)).*(logY-mean(logY)))/denOLS;
                else
                    bStart = 0;
                end

                thisFailed = false;

                for jt=1:nQR
                    tau = tauAsc(jt);
                    beta = qr_scalar_profile_warm(logY,Xc,tau,bStart);

                    if any(~isfinite(beta))
                        thisFailed = true;
                        break;
                    end

                    betaMat(:,jt)=beta;
                    bStart=beta(2);
                end

                if thisFailed
                    failed(rep)=true;
                    continue;
                end

                logQhat = Xmat*betaMat;

                if doRearrangement
                    logQhat = sort(logQhat,2,'ascend');
                end

                % ----- Stage 3: conditional Hill estimator -----
                logQbase = logQhat(:,1);
                gammaHat = sum(logQhat-logQbase,2)/(k-j0);
                gammaP = mean(gammaHat);

                if ~isfinite(gammaP) || gammaP<=1e-10 || ...
                        any(~isfinite(gammaHat))
                    failed(rep)=true;
                    continue;
                end

                % ----- Wang-Li constancy statistic -----
                Tn = mean((gammaHat-gammaP).^2);
                stat = k*Tn/(gammaP^2);

                % chi-square_1 upper-tail probability
                pval = gammainc(stat/2,0.5,'upper');

                gammaPRep(rep)=gammaP;
                TnRep(rep)=Tn;
                statRep(rep)=stat;
                pvalues(rep)=pval;
                rejected(rep)=(pval<=level);
            end

            used = ~failed & isfinite(pvalues);
            nUsed = sum(used);

            N_col(row)=n;
            K_col(row)=k;
            C_col(row)=c;
            J0_col(row)=j0;
            NumQR_col(row)=nQR;
            Fail_col(row)=MC-nUsed;

            if nUsed>0
                rr = mean(rejected(used));
                Reject_col(row)=rr;
                MCSE_col(row)=sqrt(rr*(1-rr)/nUsed);
                MeanGammaP_col(row)=mean(gammaPRep(used));
                SdGammaP_col(row)=std(gammaPRep(used));
                MeanTn_col(row)=mean(TnRep(used));
                MeanStat_col(row)=mean(statRep(used));
                SdStat_col(row)=std(statRep(used));
            else
                Reject_col(row)=NaN;
                MCSE_col(row)=NaN;
                MeanGammaP_col(row)=NaN;
                SdGammaP_col(row)=NaN;
                MeanTn_col(row)=NaN;
                MeanStat_col(row)=NaN;
                SdStat_col(row)=NaN;
            end

            replications{row}=struct( ...
                'n',n,'k',k,'c',c,'j0',j0,'tau_grid',tauAsc, ...
                'pvalue',pvalues,'reject',rejected, ...
                'gamma_pooled',gammaPRep,'Tn',TnRep, ...
                'chi2_stat',statRep,'failed',failed);

            fprintf('  rejection rate=%.4f, mean gamma_P=%.4f, failures=%d\n', ...
                Reject_col(row),MeanGammaP_col(row),Fail_col(row));
        end
    end
end

toc;

%% Save
results = table( ...
    N_col,K_col,C_col,J0_col,NumQR_col,Reject_col,MCSE_col, ...
    MeanGammaP_col,SdGammaP_col,MeanTn_col,MeanStat_col,SdStat_col, ...
    Fail_col,Level_col, ...
    'VariableNames', { ...
    'n','k','c','j0_floor_n_eta','quantile_regressions_per_rep', ...
    'rejection_rate','mcse','mean_gamma_pooled','sd_gamma_pooled', ...
    'mean_Tn','mean_chi2_stat','sd_chi2_stat','failures','nominal_level'});

disp(results);

writetable(results,'DGP2_WangLi2013_summary.xlsx','Sheet','Summary');
writetable(results,'DGP2_WangLi2013_results.csv');

settings = table(MC,level,etaTrunc,numWorkers,baseSeed,doRearrangement, ...
    'VariableNames', {'MC_replications','nominal_level','eta', ...
    'parallel_workers','base_seed','rearrangement'});
writetable(settings,'DGP2_WangLi2013_summary.xlsx','Sheet','Settings');

save('DGP2_WangLi2013_replications.mat', ...
    'replications','results','settings','nList','kByN','cList', ...
    'MC','level','etaTrunc','numWorkers','baseSeed', ...
    'doRearrangement','-v7.3');

%% Plot
figure;
hold on;
for in=1:numel(nList)
    for ik=1:size(kByN,2)
        n=nList(in);
        k=kByN(in,ik);
        rows=(results.n==n)&(results.k==k);
        plot(results.c(rows),results.rejection_rate(rows),'-o', ...
            'LineWidth',1.3,'DisplayName',sprintf('n=%d, k=%d',n,k));
    end
end
yline(level,'--','Nominal 5%','LineWidth',1.1);
xlabel('Local-alternative parameter c');
ylabel('Rejection probability');
title('DGP 2: Wang-Li (2013) power');
legend('Location','best');
grid on;
hold off;

end


function beta = qr_scalar_profile_warm(y,x,tau,bCenter)

y=y(:);
x=x(:);

if ~isfinite(bCenter)
    bCenter=0;
end

sx=std(x);
sy=std(y);
slopeScale=sy/max(sx,1e-8);
radius=2.5*max([0.25,slopeScale,abs(bCenter)/2]);

opts=optimset('Display','off','TolX',5e-7, ...
    'MaxIter',120,'MaxFunEvals',240);

obj=@(b1) qr_profile_loss(b1,y,x,tau);

for attempt=1:4
    lb=bCenter-radius;
    ub=bCenter+radius;
    [b1,~]=fminbnd(obj,lb,ub,opts);

    span=ub-lb;
    if (b1-lb)>=0.005*span && (ub-b1)>=0.005*span
        break;
    end
    radius=2*radius;
end

r=y-b1*x;
b0=empirical_tau_quantile(r,tau);
beta=[b0;b1];

end


function loss=qr_profile_loss(b1,y,x,tau)

r=y-b1*x;
b0=empirical_tau_quantile(r,tau);
u=r-b0;
loss=sum(u.*(tau-(u<0)));

end


function q=empirical_tau_quantile(v,tau)

v=sort(v(:));
n=numel(v);
idx=ceil(tau*n);
idx=max(1,min(n,idx));
q=v(idx);

end
