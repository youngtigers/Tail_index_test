function results = simulate_DGP3c_cubic_global_dcov()
% SIMULATE_DGP3C_CUBIC_GLOBAL_DCOV
% Monte Carlo for the GLOBAL top-k permutation dCov test under DGP 3c.
%
% DGP 3c: cubic-perturbed rotated interaction
%   X1,X2 iid ~ Unif[-1,1]
%   h0(X1,X2) = X1*X2 + 0.5*(X2^2-X1^2)
%   delta = 0.5
%   g(X1,X2) = [h0 + (delta/2)*(X1^3+X2^3)]/(1+delta)
%   alpha_theta(X) = 2*exp(theta*g(X1,X2))
%   kappa(X) = 1
%   Y = U^(-1/alpha_theta(X)).
%
% GLOBAL TEST:
%   1. Select exactly the largest k observations of Y.
%   2. Use the next order statistic w_hat = Y_(n-k:n) as threshold.
%   3. Form Z_i = log(Y_i/w_hat) for the selected observations.
%   4. Compute dCov between (X1,X2) and Z.
%   5. Calibrate by permutation of Z across the selected covariates.
%
% Because the same global threshold is used for every selected observation,
% the numerical value of w_hat cancels from pairwise distances in Z; it is
% nevertheless stored for diagnostics.
%
% Settings are aligned with the DGP 3c adaptive, K-F, and Wang-Li runs:
%   n = 2000, 5000
%   k = (100,50) for n=2000 and (200,100) for n=5000
%   theta = 0, 0.5, 1.0, 1.5
%   MC = 1000, permutations = 299, nominal level = 5%
%   base seed = 2026100814
%
% Outputs:
%   DGP3c_cubic_dcov_global_summary.xlsx
%   DGP3c_cubic_dcov_global_results.csv
%   DGP3c_cubic_dcov_global_replications.mat

%% Settings
numWorkers = 8;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local',numWorkers);
elseif pool.NumWorkers ~= numWorkers
    delete(pool);
    parpool('local',numWorkers);
end

baseSeed = 2026100814;

nList = [2000,5000];
kByN  = [100,50; ...
         200,100];
thetaList = [0,0.5,1.0,1.5];
delta = 0.5;

gMin = -121/162;
gMax = 1;

MC = 1000;
Rperm = 299;
level = 0.05;

fprintf('DGP 3c: GLOBAL top-k dCov, cubic-perturbed rotated interaction\n');
fprintf('g=[X1*X2+0.5*(X2^2-X1^2)+(delta/2)*(X1^3+X2^3)]/(1+delta), delta=%.2f\n',delta);
fprintf('MC=%d, permutations=%d\n\n',MC,Rperm);

%% Storage
nDesigns = numel(nList)*size(kByN,2)*numel(thetaList);

N_col        = zeros(nDesigns,1);
K_col        = zeros(nDesigns,1);
Theta_col    = zeros(nDesigns,1);
AlphaMin_col = zeros(nDesigns,1);
AlphaMax_col = zeros(nDesigns,1);
Reject_col   = NaN(nDesigns,1);
MCSE_col     = NaN(nDesigns,1);
MeanW_col    = NaN(nDesigns,1);
SdW_col      = NaN(nDesigns,1);
MeanStat_col = NaN(nDesigns,1);
SdStat_col   = NaN(nDesigns,1);
Perm_col     = Rperm*ones(nDesigns,1);
Level_col    = level*ones(nDesigns,1);

replications = cell(nDesigns,1);

row = 0;
tic;

for in = 1:numel(nList)
    n = nList(in);

    for ik = 1:size(kByN,2)
        k = kByN(in,ik);

        for it = 1:numel(thetaList)
            theta = thetaList(it);

            row = row + 1;
            designID = row;

            rejected = false(MC,1);
            pvalues  = NaN(MC,1);
            wHatRep  = NaN(MC,1);
            statRep  = NaN(MC,1);

            fprintf('Running n=%d, k=%d, theta=%.2f ...\n',n,k,theta);

            parfor rep = 1:MC
                rng(baseSeed + 100000*designID + rep,'twister');

                % ----- Generate DGP 3c -----
                X1 = 2*rand(n,1)-1;
                X2 = 2*rand(n,1)-1;
                U  = rand(n,1);
                U(U==0) = realmin;

                h0 = X1.*X2 + 0.5.*(X2.^2-X1.^2);
                g  = (h0 + (delta/2).*(X1.^3+X2.^3))./(1+delta);
                alpha = 2.*exp(theta.*g);
                Y = U.^(-1./alpha);

                % ----- Select exactly the top k observations -----
                [Ydesc,order] = sort(Y,'descend');
                wHat = Ydesc(k+1);
                tailIdx = order(1:k);

                Xe = [X1(tailIdx),X2(tailIdx)];
                Ze = log(Y(tailIdx)./wHat);

                % ----- dCov permutation test -----
                [pval,statObs] = dcov_permutation_pvalue_multivariate(Xe,Ze,Rperm);

                pvalues(rep)  = pval;
                rejected(rep) = (pval <= level);
                wHatRep(rep)  = wHat;
                statRep(rep)  = statObs;
            end

            rr = mean(rejected);

            N_col(row)        = n;
            K_col(row)        = k;
            Theta_col(row)    = theta;
            AlphaMin_col(row) = 2*exp(theta*gMin);
            AlphaMax_col(row) = 2*exp(theta*gMax);
            Reject_col(row)   = rr;
            MCSE_col(row)     = sqrt(rr*(1-rr)/MC);
            MeanW_col(row)    = mean(wHatRep);
            SdW_col(row)      = std(wHatRep);
            MeanStat_col(row) = mean(statRep);
            SdStat_col(row)   = std(statRep);

            replications{row} = struct( ...
                'n',n,'k',k,'theta',theta, ...
                'pvalue',pvalues,'reject',rejected, ...
                'w_hat',wHatRep,'dcov2_stat',statRep);

            fprintf('  rejection rate=%.4f, MCSE=%.4f\n', ...
                Reject_col(row),MCSE_col(row));
        end
    end
end

toc;

%% Save
results = table( ...
    N_col,K_col,Theta_col,AlphaMin_col,AlphaMax_col, ...
    Reject_col,MCSE_col,MeanW_col,SdW_col,MeanStat_col,SdStat_col, ...
    Perm_col,Level_col, ...
    'VariableNames', { ...
    'n','k','theta','alpha_min','alpha_max', ...
    'rejection_rate','mcse','mean_w_hat','sd_w_hat', ...
    'mean_dcov2_stat','sd_dcov2_stat','permutations','nominal_level'});

disp(results);

writetable(results,'DGP3c_cubic_dcov_global_summary.xlsx','Sheet','Summary');
writetable(results,'DGP3c_cubic_dcov_global_results.csv');

settings = table(MC,Rperm,level,numWorkers,baseSeed,delta, ...
    'VariableNames', {'MC_replications','permutations','nominal_level', ...
    'parallel_workers','base_seed','delta'});
writetable(settings,'DGP3c_cubic_dcov_global_summary.xlsx','Sheet','Settings');

save('DGP3c_cubic_dcov_global_replications.mat', ...
    'replications','results','settings','nList','kByN','thetaList', ...
    'MC','Rperm','level','numWorkers','baseSeed','delta','gMin','gMax','-v7.3');

%% Plot
figure;
hold on;
for in = 1:numel(nList)
    for ik = 1:size(kByN,2)
        n = nList(in);
        k = kByN(in,ik);
        rows = (results.n==n) & (results.k==k);
        plot(results.theta(rows),results.rejection_rate(rows),'-o', ...
            'LineWidth',1.3,'DisplayName',sprintf('n=%d, k=%d',n,k));
    end
end
yline(level,'--','Nominal 5%','LineWidth',1.1);
xlabel('\theta');
ylabel('Rejection probability');
title('DGP 3c: global top-k dCov power');
legend('Location','best');
grid on;
hold off;

end


function [pval,statObs] = dcov_permutation_pvalue_multivariate(X,Z,Rperm)
% Permutation p-value for squared sample distance covariance.
% X may be multivariate; Z is scalar here.

X = double(X);
Z = double(Z(:));

m = size(X,1);
if numel(Z) ~= m || m < 2
    error('X and Z must have the same number of observations, at least 2.');
end

% Euclidean pairwise distances for X.
xx = sum(X.^2,2);
DX2 = max(xx + xx.' - 2*(X*X.'),0);
DX = sqrt(DX2);

% Absolute distances for scalar Z.
DZ = abs(Z-Z.');

% Double center both distance matrices.
A = double_center_distance(DX);
B = double_center_distance(DZ);

statObs = mean(A(:).*B(:));

% Under a permutation of Z, the centered distance matrix is just B(p,p).
geCount = 0;
for b = 1:Rperm
    p = randperm(m);
    Bp = B(p,p);
    statPerm = mean(A(:).*Bp(:));
    geCount = geCount + (statPerm >= statObs);
end

pval = (1 + geCount)/(Rperm + 1);

end


function C = double_center_distance(D)
rowMean = mean(D,2);
colMean = mean(D,1);
grandMean = mean(D(:));
C = D - rowMean - colMean + grandMean;
end
