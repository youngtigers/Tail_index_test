function results = simulate_DGP3_local_J23()
% SIMULATE_DGP3_LOCAL_J23
% DGP 3: proposed dCov test with locally balanced X-windows.
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
% LOCAL-WINDOW PROCEDURE:
%   Run two specifications:
%      B = 2 empirical-quantile groups for EACH covariate -> 2 x 2 = 4 cells.
%      B = 3 empirical-quantile groups for EACH covariate -> 3 x 3 = 9 cells.
%
%   For each B:
%   1. Divide X1 into B empirical equal-frequency groups.
%   2. Divide X2 into B empirical equal-frequency groups.
%   3. Their Cartesian product gives B^2 local windows.
%   4. Allocate exactly k retained observations across the B^2 windows as
%      evenly as possible. Each cell gets floor(k/B^2), and any remainder
%      is allocated one each to the cells with the largest sample sizes.
%   5. Within cell j, let w_j be the next local order statistic and form
%         Z_i^* = log(Y_i/w_j)
%      for the selected observations.
%   6. Pool all selected (X1,X2,Z*) observations across cells.
%   7. Compute multivariate dCov between X=(X1,X2) and Z*, and calibrate
%      by GLOBAL permutation of the pooled Z* values.
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
%   DGP3_dcov_local_J23_summary.xlsx
%   DGP3_dcov_local_J23_results.csv
%   DGP3_dcov_local_J23_replications.mat

%% User settings
numWorkers = 8;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local', numWorkers);
elseif pool.NumWorkers ~= numWorkers
    delete(pool);
    parpool('local', numWorkers);
end

baseSeed = 20260915;

nList     = [2000,5000];
kByN      = [100, 50; ...
             200,100];
thetaList = [0,0.15,0.30,0.45];

binsPerCovariateList = [2,3];

MC    = 1000;
Rperm = 299;
level = 0.05;

fprintf('DGP 3: proposed dCov test with local X-windows\n');
fprintf('Bins per covariate: 2 or 3, giving 4 or 9 total cells\n');
fprintf('alpha_theta(X)=2*exp(theta*X1), kappa(X)=exp(X2/2)\n');
fprintf('MC=%d, permutations=%d\n\n',MC,Rperm);

%% Storage
nDesigns = numel(binsPerCovariateList) * numel(nList) * ...
           size(kByN,2) * numel(thetaList);

N_col        = zeros(nDesigns,1);
K_col        = zeros(nDesigns,1);
Bins_col     = zeros(nDesigns,1);
J_col        = zeros(nDesigns,1);
Theta_col    = zeros(nDesigns,1);
AlphaMin_col = zeros(nDesigns,1);
AlphaMax_col = zeros(nDesigns,1);
Reject_col   = zeros(nDesigns,1);
MCSE_col     = zeros(nDesigns,1);
AvgW_col     = zeros(nDesigns,1);
SdAvgW_col   = zeros(nDesigns,1);
MinW_col     = zeros(nDesigns,1);
MaxW_col     = zeros(nDesigns,1);
MinCellN_col = zeros(nDesigns,1);
MaxCellN_col = zeros(nDesigns,1);
MinCellK_col = zeros(nDesigns,1);
MaxCellK_col = zeros(nDesigns,1);
Perm_col     = Rperm*ones(nDesigns,1);
Level_col    = level*ones(nDesigns,1);

replications = cell(nDesigns,1);

row = 0;
tic;

for ib = 1:numel(binsPerCovariateList)
    B = binsPerCovariateList(ib);
    J = B^2;

    for in = 1:numel(nList)
        n = nList(in);

        for ik = 1:size(kByN,2)
            k = kByN(in,ik);

            baseK = floor(k/J);
            remainderK = mod(k,J);

            for it = 1:numel(thetaList)
                theta = thetaList(it);

                row = row+1;
                designID = row;

                rejected     = false(MC,1);
                pvalues      = NaN(MC,1);
                avgWHatRep   = NaN(MC,1);
                minWHatRep   = NaN(MC,1);
                maxWHatRep   = NaN(MC,1);
                minCellNRep  = NaN(MC,1);
                maxCellNRep  = NaN(MC,1);
                minCellKRep  = NaN(MC,1);
                maxCellKRep  = NaN(MC,1);
                localWHatRep = NaN(MC,J);
                localKRep    = NaN(MC,J);
                cellNRep     = NaN(MC,J);

                fprintf(['Running B=%d per covariate (%d cells), ' ...
                         'n=%d, k=%d, theta=%.2f ...\n'], ...
                        B,J,n,k,theta);

                parfor rep = 1:MC
                    rng(baseSeed + 100000*designID + rep,'twister');

                    % ----- Generate DGP 3 -----
                    X1 = 2*rand(n,1)-1;
                    X2 = 2*rand(n,1)-1;
                    U  = rand(n,1);
                    U(U==0) = realmin;

                    alpha = 2.*exp(theta.*X1);
                    kappa = exp(X2/2);
                    Y = (kappa./U).^(1./alpha);

                    % -----------------------------------------------------
                    % B x B empirical-quantile partition of (X1,X2)
                    % -----------------------------------------------------
                    bin1 = empirical_equal_frequency_bins(X1,B);
                    bin2 = empirical_equal_frequency_bins(X2,B);

                    cellID = (bin1-1)*B + bin2;

                    cellN = zeros(J,1);
                    for j = 1:J
                        cellN(j) = sum(cellID==j);
                    end

                    % -----------------------------------------------------
                    % Allocate exactly k selected observations across cells.
                    % -----------------------------------------------------
                    kCell = baseK*ones(J,1);

                    if remainderK>0
                        [~,largeCells] = sort(cellN,'descend');
                        kCell(largeCells(1:remainderK)) = ...
                            kCell(largeCells(1:remainderK)) + 1;
                    end

                    if any(cellN <= kCell)
                        error('A local cell has too few observations for its local threshold.');
                    end

                    Xe = NaN(k,2);
                    Ze = NaN(k,1);
                    wLocal = NaN(J,1);

                    pos = 0;

                    for j = 1:J
                        idxCell = find(cellID==j);

                        X1j = X1(idxCell);
                        X2j = X2(idxCell);
                        Yj  = Y(idxCell);
                        kj  = kCell(j);

                        [YjDesc,orderLocal] = sort(Yj,'descend');

                        % Local threshold = next order statistic.
                        wj = YjDesc(kj+1);
                        localTop = orderLocal(1:kj);

                        sel = (pos+1):(pos+kj);

                        Xe(sel,1) = X1j(localTop);
                        Xe(sel,2) = X2j(localTop);
                        Ze(sel)   = log(Yj(localTop)./wj);

                        wLocal(j) = wj;
                        pos = pos+kj;
                    end

                    if pos~=k
                        error('Local selection did not produce exactly k observations.');
                    end

                    % ----- Global dCov permutation test on pooled sample -----
                    pval = dcov_permutation_pvalue_multivariate(Xe,Ze,Rperm);

                    pvalues(rep) = pval;
                    rejected(rep) = (pval<=level);

                    localWHatRep(rep,:) = wLocal.';
                    localKRep(rep,:)    = kCell.';
                    cellNRep(rep,:)     = cellN.';

                    avgWHatRep(rep)  = mean(wLocal);
                    minWHatRep(rep)  = min(wLocal);
                    maxWHatRep(rep)  = max(wLocal);
                    minCellNRep(rep) = min(cellN);
                    maxCellNRep(rep) = max(cellN);
                    minCellKRep(rep) = min(kCell);
                    maxCellKRep(rep) = max(kCell);
                end

                rr = mean(rejected);

                N_col(row)        = n;
                K_col(row)        = k;
                Bins_col(row)     = B;
                J_col(row)        = J;
                Theta_col(row)    = theta;
                AlphaMin_col(row) = 2*exp(-abs(theta));
                AlphaMax_col(row) = 2*exp(abs(theta));
                Reject_col(row)   = rr;
                MCSE_col(row)     = sqrt(rr*(1-rr)/MC);

                AvgW_col(row)     = mean(avgWHatRep);
                SdAvgW_col(row)   = std(avgWHatRep);
                MinW_col(row)     = mean(minWHatRep);
                MaxW_col(row)     = mean(maxWHatRep);
                MinCellN_col(row) = mean(minCellNRep);
                MaxCellN_col(row) = mean(maxCellNRep);
                MinCellK_col(row) = mean(minCellKRep);
                MaxCellK_col(row) = mean(maxCellKRep);

                replications{row} = struct( ...
                    'n',n,'k',k, ...
                    'bins_per_covariate',B, ...
                    'windows',J, ...
                    'theta',theta, ...
                    'pvalue',pvalues, ...
                    'reject',rejected, ...
                    'local_w_hat',localWHatRep, ...
                    'local_k',localKRep, ...
                    'cell_n',cellNRep, ...
                    'average_local_w_hat',avgWHatRep, ...
                    'minimum_local_w_hat',minWHatRep, ...
                    'maximum_local_w_hat',maxWHatRep);

                fprintf('  rejection rate=%.4f, MCSE=%.4f\n', ...
                    Reject_col(row),MCSE_col(row));
            end
        end
    end
end

toc;

%% Save
results = table( ...
    N_col,K_col,Bins_col,J_col,Theta_col,AlphaMin_col,AlphaMax_col, ...
    Reject_col,MCSE_col, ...
    AvgW_col,SdAvgW_col,MinW_col,MaxW_col, ...
    MinCellN_col,MaxCellN_col,MinCellK_col,MaxCellK_col, ...
    Perm_col,Level_col, ...
    'VariableNames', { ...
    'n','k','bins_per_covariate','windows','theta','alpha_min','alpha_max', ...
    'rejection_rate','mcse', ...
    'mean_average_local_w_hat','sd_average_local_w_hat', ...
    'mean_min_local_w_hat','mean_max_local_w_hat', ...
    'mean_min_cell_n','mean_max_cell_n', ...
    'mean_min_k_per_cell','mean_max_k_per_cell', ...
    'permutations','nominal_level'});

disp(results);

writetable(results,'DGP3_dcov_local_J23_summary.xlsx','Sheet','Summary');
writetable(results,'DGP3_dcov_local_J23_results.csv');

settings = table( ...
    binsPerCovariateList(:), binsPerCovariateList(:).^2, ...
    repmat(MC,numel(binsPerCovariateList),1), ...
    repmat(Rperm,numel(binsPerCovariateList),1), ...
    repmat(level,numel(binsPerCovariateList),1), ...
    repmat(numWorkers,numel(binsPerCovariateList),1), ...
    repmat(baseSeed,numel(binsPerCovariateList),1), ...
    'VariableNames', { ...
    'bins_per_covariate','windows','MC_replications','permutations', ...
    'nominal_level','parallel_workers','base_seed'});

writetable(settings,'DGP3_dcov_local_J23_summary.xlsx','Sheet','Settings');

save('DGP3_dcov_local_J23_replications.mat', ...
    'replications','results','settings','nList','kByN','thetaList', ...
    'binsPerCovariateList','MC','Rperm','level', ...
    'numWorkers','baseSeed','-v7.3');

%% Plot
figure;
hold on;

for ib = 1:numel(binsPerCovariateList)
    B = binsPerCovariateList(ib);
    for in = 1:numel(nList)
        for ik = 1:size(kByN,2)
            n = nList(in);
            k = kByN(in,ik);

            rows = (results.bins_per_covariate==B) & ...
                   (results.n==n) & (results.k==k);

            plot(results.theta(rows),results.rejection_rate(rows),'-o', ...
                'LineWidth',1.3, ...
                'DisplayName',sprintf('B=%d, n=%d, k=%d',B,n,k));
        end
    end
end

yline(level,'--','Nominal 5%','LineWidth',1.1);
xlabel('\theta');
ylabel('Rejection probability');
title('DGP 3: dCov power, B=2 or 3 bins per covariate');
legend('Location','best');
grid on;
hold off;

end


function bin = empirical_equal_frequency_bins(X,B)
% Assign observations to B empirical equal-frequency bins using ranks.

X = X(:);
n = numel(X);

[~,ord] = sort(X,'ascend');

bin = zeros(n,1);

for r = 1:n
    b = ceil(B*r/n);
    b = max(1,min(B,b));
    bin(ord(r)) = b;
end

end


function pval = dcov_permutation_pvalue_multivariate(X,Z,Rperm)
% Global permutation test after pooling all local-window exceedances.

Z = Z(:);
m = size(X,1);

A = centered_euclidean_distance_matrix(X);
B = centered_euclidean_distance_matrix(Z);

Tobs = sum(A.*B,'all')/(m^2);

nGreaterEqual = 0;

for r = 1:Rperm
    pi = randperm(m);
    Bpi = B(pi,pi);

    Tperm = sum(A.*Bpi,'all')/(m^2);

    nGreaterEqual = nGreaterEqual + (Tperm>=Tobs);
end

pval = (1+nGreaterEqual)/(Rperm+1);

end


function A = centered_euclidean_distance_matrix(V)

if isvector(V)
    V = V(:);
end

sqNorm = sum(V.^2,2);
D2 = sqNorm + sqNorm' - 2*(V*V');
D2 = max(D2,0);
D = sqrt(D2);

rowMean = mean(D,2);
colMean = mean(D,1);
grandMean = mean(D,'all');

A = D-rowMean-colMean+grandMean;

end
