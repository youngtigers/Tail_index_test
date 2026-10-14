function results = simulate_DGP3_WangLi2013()
% SIMULATE_DGP3_WANGLI2013
% DGP 3: Wang and Li (2013) constant-EVI test with two covariates.
%
% DGP:
%   X1,X2 iid ~ Unif[-1,1]
%   alpha_theta(X)=2*exp(theta*X1)
%   kappa(X)=exp(X2/2)
%   Y={kappa(X)/U}^{1/alpha_theta(X)}.
%
% The EVI is gamma_theta(X)=0.5*exp(-theta*X1).
%
% Under theta=0:
%   log(Y)=X2/4-0.5*log(U),
% so lambda=0 is the population-correct transformation and the transformed
% conditional quantiles are exactly linear in (X1,X2).
%
% Under theta>0 the transformed quantile surface is nonlinear. Thus DGP 3
% intentionally examines nonlinear/multivariate heterogeneity.
%
% Implementation:
%   - lambda=0
%   - eta=0.1
%   - upper conditional QR of log(Y) on (1,X1,X2)
%   - rearrangement across quantile levels
%   - conditional Hill estimator gamma_hat(x)
%   - T_n=n^{-1}sum_i{gamma_hat(X_i)-gamma_hat_P}^2
%
% Since lambda=0 implies transformed EVI gamma_0^*=0, Wang-Li Corollary
% 3.1 gives, after centering both covariates,
%
%   k*T_n/gamma_hat_P^2 ==> chi-square_{p-1}=chi-square_2,
%
% because p=3 includes the intercept.
%
% Settings: MC=1000.
%
% Outputs:
%   DGP3_WangLi2013_summary.xlsx
%   DGP3_WangLi2013_results.csv
%   DGP3_WangLi2013_replications.mat

%% Settings
numWorkers=48;
pool=gcp('nocreate');
if isempty(pool)
    parpool('local',numWorkers);
elseif pool.NumWorkers~=numWorkers
    delete(pool);
    parpool('local',numWorkers);
end

baseSeed=20260914;

nList=[2000,5000];
kByN=[100,50; ...
      200,100];
thetaList=[0,0.15,0.30,0.45];

MC=1000;
level=0.05;
etaTrunc=0.1;
doRearrangement=true;

fprintf('DGP 3: Wang and Li (2013) constant-EVI test\n');
fprintf('Two covariates; chi-square_2 Corollary 3.1 calibration\n');
fprintf('MC=%d, eta=%.2f\n\n',MC,etaTrunc);

%% Storage
nDesigns=numel(nList)*size(kByN,2)*numel(thetaList);

N_col=zeros(nDesigns,1);
K_col=zeros(nDesigns,1);
Theta_col=zeros(nDesigns,1);
J0_col=zeros(nDesigns,1);
NumQR_col=zeros(nDesigns,1);
Reject_col=zeros(nDesigns,1);
MCSE_col=zeros(nDesigns,1);
MeanGammaP_col=zeros(nDesigns,1);
SdGammaP_col=zeros(nDesigns,1);
MeanTn_col=zeros(nDesigns,1);
MeanStat_col=zeros(nDesigns,1);
SdStat_col=zeros(nDesigns,1);
Fail_col=zeros(nDesigns,1);
Level_col=level*ones(nDesigns,1);

replications=cell(nDesigns,1);

row=0;
tic;

for in=1:numel(nList)
    n=nList(in);

    for ik=1:size(kByN,2)
        k=kByN(in,ik);

        j0=floor(n^etaTrunc);
        j0=max(j0,1);

        if k<=j0
            error('Need k > floor(n^eta).');
        end

        jAsc=(k:-1:j0).';
        tauAsc=(n-jAsc)/(n+1);
        nQR=numel(tauAsc);

        for it=1:numel(thetaList)
            theta=thetaList(it);

            row=row+1;
            designID=row;

            rejected=false(MC,1);
            pvalues=NaN(MC,1);
            gammaPRep=NaN(MC,1);
            TnRep=NaN(MC,1);
            statRep=NaN(MC,1);
            failed=false(MC,1);

            fprintf('Running n=%d, k=%d, theta=%.2f (%d QR fits/rep) ...\n', ...
                n,k,theta,nQR);

            parfor rep=1:MC
                rng(baseSeed+100000*designID+rep,'twister');

                % ----- Generate DGP 3 -----
                X1=2*rand(n,1)-1;
                X2=2*rand(n,1)-1;

                U=rand(n,1);
                U(U==0)=realmin;

                alpha=2.*exp(theta.*X1);
                kappa=exp(X2/2);
                Y=(kappa./U).^(1./alpha);
                logY=log(Y);

                % Center both non-intercept regressors for Corollary 3.1.
                X1c=X1-mean(X1);
                X2c=X2-mean(X2);
                X=[X1c,X2c];
                Xmat=[ones(n,1),X];

                % ----- Stage 2: upper-tail QR on log(Y) -----
                betaMat=NaN(3,nQR);

                betaOLS=Xmat\logY;
                slopeStart=betaOLS(2:end);

                thisFailed=false;

                for jt=1:nQR
                    tau=tauAsc(jt);

                    beta=qr_multivariate_profile(logY,X,tau,slopeStart);

                    if any(~isfinite(beta))
                        thisFailed=true;
                        break;
                    end

                    betaMat(:,jt)=beta;
                    slopeStart=beta(2:end);
                end

                if thisFailed
                    failed(rep)=true;
                    continue;
                end

                logQhat=Xmat*betaMat;

                if doRearrangement
                    logQhat=sort(logQhat,2,'ascend');
                end

                % ----- Stage 3: conditional Hill estimator -----
                logQbase=logQhat(:,1);
                gammaHat=sum(logQhat-logQbase,2)/(k-j0);
                gammaP=mean(gammaHat);

                if ~isfinite(gammaP) || gammaP<=1e-10 || ...
                        any(~isfinite(gammaHat))
                    failed(rep)=true;
                    continue;
                end

                % ----- Wang-Li constancy test -----
                Tn=mean((gammaHat-gammaP).^2);

                % p=3 => p-1=2.
                stat=k*Tn/(gammaP^2);
                pval=gammainc(stat/2,1,'upper');

                gammaPRep(rep)=gammaP;
                TnRep(rep)=Tn;
                statRep(rep)=stat;
                pvalues(rep)=pval;
                rejected(rep)=(pval<=level);
            end

            used=~failed&isfinite(pvalues);
            nUsed=sum(used);

            N_col(row)=n;
            K_col(row)=k;
            Theta_col(row)=theta;
            J0_col(row)=j0;
            NumQR_col(row)=nQR;
            Fail_col(row)=MC-nUsed;

            if nUsed>0
                rr=mean(rejected(used));
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
                'n',n,'k',k,'theta',theta,'j0',j0,'tau_grid',tauAsc, ...
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
results=table( ...
    N_col,K_col,Theta_col,J0_col,NumQR_col,Reject_col,MCSE_col, ...
    MeanGammaP_col,SdGammaP_col,MeanTn_col,MeanStat_col,SdStat_col, ...
    Fail_col,Level_col, ...
    'VariableNames', { ...
    'n','k','theta','j0_floor_n_eta','quantile_regressions_per_rep', ...
    'rejection_rate','mcse','mean_gamma_pooled','sd_gamma_pooled', ...
    'mean_Tn','mean_chi2_stat','sd_chi2_stat','failures','nominal_level'});

disp(results);

writetable(results,'DGP3_WangLi2013_summary.xlsx','Sheet','Summary');
writetable(results,'DGP3_WangLi2013_results.csv');

settings=table(MC,level,etaTrunc,numWorkers,baseSeed,doRearrangement, ...
    'VariableNames', {'MC_replications','nominal_level','eta', ...
    'parallel_workers','base_seed','rearrangement'});
writetable(settings,'DGP3_WangLi2013_summary.xlsx','Sheet','Settings');

save('DGP3_WangLi2013_replications.mat', ...
    'replications','results','settings','nList','kByN','thetaList', ...
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
        plot(results.theta(rows),results.rejection_rate(rows),'-o', ...
            'LineWidth',1.3,'DisplayName',sprintf('n=%d, k=%d',n,k));
    end
end
yline(level,'--','Nominal 5%','LineWidth',1.1);
xlabel('\theta');
ylabel('Rejection probability');
title('DGP 3: Wang-Li (2013) power');
legend('Location','best');
grid on;
hold off;

end


function beta=qr_multivariate_profile(y,X,tau,slopeStart)
% Check-loss QR with intercept profiled out.
% X contains regressors only. For fixed slopes, the optimal intercept is
% the empirical tau-quantile of y-X*b. The remaining two-dimensional
% convex profile problem is solved by fminsearch with warm starts.

y=y(:);
[n,d]=size(X);

if nargin<4 || numel(slopeStart)~=d || any(~isfinite(slopeStart))
    Z=[ones(n,1),X];
    bOLS=Z\y;
    slopeStart=bOLS(2:end);
end

slopeStart=slopeStart(:);

opts=optimset('Display','off','TolX',2e-6,'TolFun',1e-6, ...
    'MaxIter',180,'MaxFunEvals',450);

obj=@(b) qr_profile_loss_multi(b,y,X,tau);

[bSlope,~,exitflag]=fminsearch(obj,slopeStart,opts);

if exitflag==0
    opts2=optimset(opts,'MaxIter',300,'MaxFunEvals',750);
    [bSlope,~,~]=fminsearch(obj,bSlope,opts2);
end

r=y-X*bSlope;
b0=empirical_tau_quantile(r,tau);
beta=[b0;bSlope(:)];

end


function loss=qr_profile_loss_multi(b,y,X,tau)

r=y-X*b(:);
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
