function results = simulate_DGP3b_KF_Ltest()
% SIMULATE_DGP3B_KF_LTEST
% Pilot Monte Carlo for Kinsvater-Fried (2017) under a PURE INTERACTION
% tail-index alternative.
%
% DGP 3b:
%   X1,X2 iid ~ Unif[-1,1]
%   alpha_theta(X)=2*exp(theta*X1*X2)
%   kappa(X)=1
%   Y=U^(-1/alpha_theta(X)).
%
% The EVI is
%
%   gamma_theta(X)=0.5*exp(-theta*X1*X2).
%
% Under theta>0, neither X1 nor X2 has a first-order linear main effect.
% The Kinsvater-Fried L-test below intentionally uses only the two main
% effects, exactly as a researcher would if no interaction were specified.
%
% Implementation:
%   - target tail size k
%   - p_kn=(n-k)/(n+1)
%   - lambda=0 (log transformation)
%   - conditional threshold QR of log(Y) on (1,X1,X2)
%   - 20 exceedance QR probabilities on [0.5,0.975]
%   - optimal L-weights
%   - joint Wald test H0: eta1=eta2=0
%
% Pilot settings:
%   MC=100
%
% Outputs:
%   DGP3b_KF_Ltest_summary.xlsx
%   DGP3b_KF_Ltest_results.csv
%   DGP3b_KF_Ltest_replications.mat

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
thetaList=[0,0.5,1.0,1.5];

MC=1000;
level=0.05;

ell=20;
pGrid=linspace(0.5,1-1/40,ell).';

A=kf_A_matrix(pGrid);
oneVec=ones(ell,1);
if rcond(A)>1e-12
    Ainv1=A\oneVec;
else
    Ainv1=pinv(A)*oneVec;
end
wOpt=Ainv1/(oneVec'*Ainv1);
aWeight=wOpt'*A*wOpt;

fprintf('DGP 3b: Kinsvater-Fried L-test, pure interaction alternative\n');
fprintf('Joint test of X1 and X2 main-effect EVI slopes\n');
fprintf('MC=%d\n\n',MC);

%% Storage
nDesigns=numel(nList)*size(kByN,2)*numel(thetaList);

N_col=zeros(nDesigns,1);
K_col=zeros(nDesigns,1);
Theta_col=zeros(nDesigns,1);
Reject_col=zeros(nDesigns,1);
MCSE_col=zeros(nDesigns,1);
MeanM_col=zeros(nDesigns,1);
SdM_col=zeros(nDesigns,1);
MeanWald_col=zeros(nDesigns,1);
SdWald_col=zeros(nDesigns,1);
MeanEta0_col=zeros(nDesigns,1);
MeanEta1_col=zeros(nDesigns,1);
MeanEta2_col=zeros(nDesigns,1);
Fail_col=zeros(nDesigns,1);
Level_col=level*ones(nDesigns,1);

replications=cell(nDesigns,1);

row=0;
tic;

for in=1:numel(nList)
    n=nList(in);

    for ik=1:size(kByN,2)
        k=kByN(in,ik);

        for it=1:numel(thetaList)
            theta=thetaList(it);

            row=row+1;
            designID=row;

            rejected=false(MC,1);
            pvalues=NaN(MC,1);
            waldRep=NaN(MC,1);
            mRep=NaN(MC,1);
            eta0Rep=NaN(MC,1);
            eta1Rep=NaN(MC,1);
            eta2Rep=NaN(MC,1);
            failed=false(MC,1);

            fprintf('Running n=%d, k=%d, theta=%.2f ...\n',n,k,theta);

            parfor rep=1:MC
                rng(baseSeed+100000*designID+rep,'twister');

                % ----- Generate DGP 3b -----
                X1=2*rand(n,1)-1;
                X2=2*rand(n,1)-1;
                X=[X1,X2];

                U=rand(n,1);
                U(U==0)=realmin;

                alpha=2.*exp(theta.*X1.*X2);
                Y=U.^(-1./alpha);
                logY=log(Y);

                % ----- KF conditional threshold -----
                pkn=(n-k)/(n+1);

                Z0=[ones(n,1),X];
                betaOLS=Z0\logY;
                betaThresh=qr_multivariate_profile(logY,X,pkn,betaOLS(2:end));

                logUhat=[ones(n,1),X]*betaThresh;
                uhat=exp(logUhat);

                idx=(Y>uhat);
                Xe=X(idx,:);
                Ze=Y(idx)./uhat(idx);

                good=all(isfinite(Xe),2)&isfinite(Ze)&(Ze>1);
                Xe=Xe(good,:);
                Ze=Ze(good);

                m=numel(Ze);
                mRep(rep)=m;

                if m<35
                    failed(rep)=true;
                    continue;
                end

                % ----- KF L-estimator -----
                logZ=log(Ze);
                etaByP=NaN(3,ell);

                Zols=[ones(m,1),Xe];
                betaStart=Zols\logZ;
                slopeStart=betaStart(2:end);

                thisFailed=false;

                for jp=1:ell
                    pp=pGrid(jp);
                    betaP=qr_multivariate_profile(logZ,Xe,pp,slopeStart);

                    if any(~isfinite(betaP))
                        thisFailed=true;
                        break;
                    end

                    etaP=-betaP/log(1-pp);
                    etaByP(:,jp)=etaP;
                    slopeStart=betaP(2:end);
                end

                if thisFailed
                    failed(rep)=true;
                    continue;
                end

                etaL=etaByP*wOpt;

                eta0Rep(rep)=etaL(1);
                eta1Rep(rep)=etaL(2);
                eta2Rep(rep)=etaL(3);

                Xmat=[ones(m,1),Xe];
                gammaHat=Xmat*etaL;

                if any(~isfinite(gammaHat)) || any(gammaHat<=1e-8)
                    failed(rep)=true;
                    continue;
                end

                Jhat=(Xmat'*Xmat)/m;

                Hhat=zeros(3,3);
                for j=1:m
                    xx=Xmat(j,:).';
                    Hhat=Hhat+(xx*xx.')/gammaHat(j);
                end
                Hhat=Hhat/m;

                if rcond(Hhat)<1e-10
                    failed(rep)=true;
                    continue;
                end

                HinvJHinv=Hhat\Jhat/Hhat;
                SigmaEta=aWeight*HinvJHinv;

                b=etaL(2:3);
                V=SigmaEta(2:3,2:3);

                if rcond(V)<1e-10 || any(~isfinite(V),'all')
                    failed(rep)=true;
                    continue;
                end

                W=m*(b'*(V\b));
                pval=gammainc(W/2,1,'upper'); % chi-square_2

                waldRep(rep)=W;
                pvalues(rep)=pval;
                rejected(rep)=(pval<=level);
            end

            used=~failed&isfinite(pvalues);
            nUsed=sum(used);

            N_col(row)=n;
            K_col(row)=k;
            Theta_col(row)=theta;
            Fail_col(row)=MC-nUsed;

            if nUsed>0
                rr=mean(rejected(used));
                Reject_col(row)=rr;
                MCSE_col(row)=sqrt(rr*(1-rr)/nUsed);
                MeanM_col(row)=mean(mRep(used));
                SdM_col(row)=std(mRep(used));
                MeanWald_col(row)=mean(waldRep(used));
                SdWald_col(row)=std(waldRep(used));
                MeanEta0_col(row)=mean(eta0Rep(used));
                MeanEta1_col(row)=mean(eta1Rep(used));
                MeanEta2_col(row)=mean(eta2Rep(used));
            else
                Reject_col(row)=NaN;
                MCSE_col(row)=NaN;
                MeanM_col(row)=NaN;
                SdM_col(row)=NaN;
                MeanWald_col(row)=NaN;
                SdWald_col(row)=NaN;
                MeanEta0_col(row)=NaN;
                MeanEta1_col(row)=NaN;
                MeanEta2_col(row)=NaN;
            end

            replications{row}=struct( ...
                'n',n,'k',k,'theta',theta, ...
                'pvalue',pvalues,'reject',rejected, ...
                'wald',waldRep,'effective_tail_n',mRep, ...
                'eta0_hat',eta0Rep,'eta1_hat',eta1Rep, ...
                'eta2_hat',eta2Rep,'failed',failed);

            fprintf('  rejection rate=%.4f, mean m=%.2f, failures=%d\n', ...
                Reject_col(row),MeanM_col(row),Fail_col(row));
        end
    end
end

toc;

%% Save
results=table( ...
    N_col,K_col,Theta_col,Reject_col,MCSE_col,MeanM_col,SdM_col, ...
    MeanWald_col,SdWald_col,MeanEta0_col,MeanEta1_col,MeanEta2_col, ...
    Fail_col,Level_col, ...
    'VariableNames', { ...
    'n','k','theta','rejection_rate','mcse','mean_effective_tail_n', ...
    'sd_effective_tail_n','mean_wald','sd_wald','mean_eta0_hat', ...
    'mean_eta1_hat','mean_eta2_hat','failures','nominal_level'});

disp(results);

writetable(results,'DGP3b_KF_Ltest_summary.xlsx','Sheet','Summary');
writetable(results,'DGP3b_KF_Ltest_results.csv');

settings=table(MC,level,numWorkers,baseSeed,ell, ...
    'VariableNames', {'MC_replications','nominal_level', ...
    'parallel_workers','base_seed','L_probabilities'});
writetable(settings,'DGP3b_KF_Ltest_summary.xlsx','Sheet','Settings');

save('DGP3b_KF_Ltest_replications.mat', ...
    'replications','results','settings','nList','kByN','thetaList', ...
    'MC','level','pGrid','wOpt','A','aWeight', ...
    'numWorkers','baseSeed','-v7.3');

end


function A=kf_A_matrix(pGrid)

pGrid=pGrid(:);
ell=numel(pGrid);
A=zeros(ell,ell);

for i=1:ell
    for j=1:ell
        pi_=pGrid(i);
        pj_=pGrid(j);
        num=min(pi_,pj_)-pi_*pj_;
        den=(1-pi_)*(1-pj_)*log(1-pi_)*log(1-pj_);
        A(i,j)=num/den;
    end
end

A=0.5*(A+A.');

end


function beta=qr_multivariate_profile(y,X,tau,slopeStart)

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
