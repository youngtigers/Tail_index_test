function results = simulate_DGP3()
% SIMULATE_DGP3
% Monte Carlo power experiment for DGP 3 using the proposed dCov test.
%
% DGP 3:
%   X1, X2 iid ~ Unif[-1,1]
%   alpha_theta(X) = 2*exp(theta*X1)
%   kappa(X)       = exp(X2/2)
%   Y              = {kappa(X)/U}^{1/alpha_theta(X)}, U~Unif(0,1)
%
% X1 affects tail thickness; X2 affects tail frequency only.
% theta=0 is the null of tail-index homogeneity.
%
% ORDER-STATISTIC IMPLEMENTATION:
%   w_hat = Y_(n-k:n), so exactly the largest k observations are used.
%   The test is dCov between X=(X1,X2) and log(Y/w_hat).
%
% Settings:
%   MC = 1000
%   Rperm = 299
%
% Tail sizes:
%   n=2000: k=100,50
%   n=5000: k=200,100
%
% theta = 0, 0.15, 0.30, 0.45
%
% Outputs:
%   DGP3_dcov_summary.xlsx
%   DGP3_dcov_results.csv
%   DGP3_dcov_replications.mat

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

nList     = [2000,5000];
kByN      = [100, 50; ...
             200,100];
thetaList = [0,0.15,0.30,0.45];

MC    = 1000;
Rperm = 299;
level = 0.05;

fprintf('DGP 3: proposed dCov test\n');
fprintf('alpha_theta(X)=2*exp(theta*X1), kappa(X)=exp(X2/2)\n');
fprintf('MC=%d, permutations=%d\n\n',MC,Rperm);

%% Storage
nDesigns = numel(nList)*size(kByN,2)*numel(thetaList);

N_col        = zeros(nDesigns,1);
K_col        = zeros(nDesigns,1);
Theta_col    = zeros(nDesigns,1);
AlphaMin_col = zeros(nDesigns,1);
AlphaMax_col = zeros(nDesigns,1);
Reject_col   = zeros(nDesigns,1);
MCSE_col     = zeros(nDesigns,1);
MeanW_col    = zeros(nDesigns,1);
SdW_col      = zeros(nDesigns,1);
Perm_col     = Rperm*ones(nDesigns,1);
Level_col    = level*ones(nDesigns,1);

replications = cell(nDesigns,1);

row = 0;
tic;

for in=1:numel(nList)
    n = nList(in);

    for ik=1:size(kByN,2)
        k = kByN(in,ik);

        for it=1:numel(thetaList)
            theta = thetaList(it);

            row = row+1;
            designID = row;

            rejected = false(MC,1);
            pvalues  = NaN(MC,1);
            wHatRep  = NaN(MC,1);

            fprintf('Running n=%d, k=%d, theta=%.2f ...\n',n,k,theta);

            parfor rep=1:MC
                rng(baseSeed+100000*designID+rep,'twister');

                % ----- Generate DGP 3 -----
                X1 = 2*rand(n,1)-1;
                X2 = 2*rand(n,1)-1;
                U  = rand(n,1);
                U(U==0) = realmin;

                alpha = 2.*exp(theta.*X1);
                kappa = exp(X2/2);
                Y = (kappa./U).^(1./alpha);

                % ----- Select exactly the top k observations -----
                [Ydesc,order] = sort(Y,'descend');
                wHat = Ydesc(k+1);
                tailIdx = order(1:k);

                Xe = [X1(tailIdx),X2(tailIdx)];
                Ze = log(Y(tailIdx)./wHat);

                % ----- dCov permutation test -----
                pval = dcov_permutation_pvalue_multivariate(Xe,Ze,Rperm);

                pvalues(rep)=pval;
                rejected(rep)=(pval<=level);
                wHatRep(rep)=wHat;
            end

            rr = mean(rejected);

            N_col(row)=n;
            K_col(row)=k;
            Theta_col(row)=theta;
            AlphaMin_col(row)=2*exp(-abs(theta));
            AlphaMax_col(row)=2*exp(abs(theta));
            Reject_col(row)=rr;
            MCSE_col(row)=sqrt(rr*(1-rr)/MC);
            MeanW_col(row)=mean(wHatRep);
            SdW_col(row)=std(wHatRep);

            replications{row}=struct( ...
                'n',n,'k',k,'theta',theta, ...
                'pvalue',pvalues,'reject',rejected,'w_hat',wHatRep);

            fprintf('  rejection rate=%.4f, MCSE=%.4f\n', ...
                Reject_col(row),MCSE_col(row));
        end
    end
end

toc;

%% Save
results = table( ...
    N_col,K_col,Theta_col,AlphaMin_col,AlphaMax_col, ...
    Reject_col,MCSE_col,MeanW_col,SdW_col,Perm_col,Level_col, ...
    'VariableNames', { ...
    'n','k','theta','alpha_min','alpha_max', ...
    'rejection_rate','mcse','mean_w_hat','sd_w_hat', ...
    'permutations','nominal_level'});

disp(results);

writetable(results,'DGP3_dcov_summary.xlsx','Sheet','Summary');
writetable(results,'DGP3_dcov_results.csv');

settings = table(MC,Rperm,level,numWorkers,baseSeed, ...
    'VariableNames', {'MC_replications','permutations','nominal_level', ...
    'parallel_workers','base_seed'});
writetable(settings,'DGP3_dcov_summary.xlsx','Sheet','Settings');

save('DGP3_dcov_replications.mat', ...
    'replications','results','settings','nList','kByN','thetaList', ...
    'MC','Rperm','level','numWorkers','baseSeed','-v7.3');

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
title('DGP 3: dCov power');
legend('Location','best');
grid on;
hold off;

end


function pval = dcov_permutation_pvalue_multivariate(X,Z,Rperm)

Z=Z(:);
m=size(X,1);

A=centered_euclidean_distance_matrix(X);
B=centered_euclidean_distance_matrix(Z);

Tobs=sum(A.*B,'all')/(m^2);

nGreaterEqual=0;
for r=1:Rperm
    pi=randperm(m);
    Bpi=B(pi,pi);
    Tperm=sum(A.*Bpi,'all')/(m^2);
    nGreaterEqual=nGreaterEqual+(Tperm>=Tobs);
end

pval=(1+nGreaterEqual)/(Rperm+1);

end


function A=centered_euclidean_distance_matrix(V)

if isvector(V)
    V=V(:);
end

sqNorm=sum(V.^2,2);
D2=sqNorm+sqNorm'-2*(V*V');
D2=max(D2,0);
D=sqrt(D2);

rowMean=mean(D,2);
colMean=mean(D,1);
grandMean=mean(D,'all');

A=D-rowMean-colMean+grandMean;

end
