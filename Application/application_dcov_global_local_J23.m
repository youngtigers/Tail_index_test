function results = application_dcov_global_local_J23(dataFile,kFracList,Rperm,level)
% APPLICATION_DCOV_GLOBAL_LOCAL_J23
% French MTPL application: global and locally balanced dCov tests.
%
% Covariates:
%   X = (BonusMalus, VehPower)
%
% Tail sizes:
%   k_1 = round(0.05*n)
%   k_2 = round(0.025*n)
% by default, where n is the number of usable positive claims.
%
% GLOBAL:
%   w = Y_(n-k:n), use top k observations, Z=log(Y/w).
%
% LOCAL, B=2:
%   2 empirical-rank groups for BonusMalus
%   x 2 empirical-rank groups for VehPower = 4 cells.
%
% LOCAL, B=3:
%   3 empirical-rank groups for BonusMalus
%   x 3 empirical-rank groups for VehPower = 9 cells.
%
% For each local partition, the total k is allocated across cells as
% evenly as possible. Any remainder is allocated one at a time to the
% cells with the largest sample sizes. In cell j, w_j is the next local
% order statistic and Z_i=log(Y_i/w_j). Selected observations are pooled
% and Z is globally permuted against X.
%
% NOTE ON TIES:
%   BonusMalus and VehPower are discrete in practice. To reproduce the
%   simulation implementation with exactly equal-frequency rank groups,
%   tied covariate values can be split across adjacent rank groups. Ties
%   are broken deterministically by the original observation index.
%
% The GLOBAL, 2x2 LOCAL, and 3x3 LOCAL procedures use the same cleaned
% application sample and the same tail sizes k.
%
% Outputs:
%   application_dcov_global_local_J23_results.xlsx
%   application_dcov_global_local_J23_results.csv

if nargin < 1 || isempty(dataFile)
    dataFile = 'freMTPL2_application.csv';
end
if nargin < 2 || isempty(kFracList)
    kFracList = [0.05,0.025];
end
if nargin < 3 || isempty(Rperm)
    Rperm = 299;
end
if nargin < 4 || isempty(level)
    level = 0.05;
end

rng(20260915,'twister');

%% Parallel pool
numWorkers = 8;
pool = gcp('nocreate');
if isempty(pool)
    parpool('local',numWorkers);
elseif pool.NumWorkers ~= numWorkers
    delete(pool);
    parpool('local',numWorkers);
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

% Standardize using the full usable application sample. This affects only
% the Euclidean scale used by dCov; empirical-rank group membership is
% unchanged by this affine transformation.
Xmean = mean(Xraw,1);
Xsd = std(Xraw,0,1);
if any(Xsd<=0)
    error('At least one covariate has zero variance.');
end
X = (Xraw-Xmean)./Xsd;

kList = round(n*kFracList);
BList = [2,3];

fprintf('\nFrench MTPL application: dCov\n');
fprintf('Usable positive claims: n=%d\n',n);
fprintf('X=(BonusMalus, VehPower)\n');
fprintf('Tail sizes: ');
fprintf('%d ',kList);
fprintf('\nProcedures: global, 2x2 local, and 3x3 local\n\n');

%% Storage
nK = numel(kList);

K_col = zeros(nK,1);
Kfrac_col = zeros(nK,1);
EquivalentQ_col = zeros(nK,1);

GlobalW_col = zeros(nK,1);
GlobalStat_col = zeros(nK,1);
GlobalP_col = zeros(nK,1);

% B=2 means 2 bins per covariate, hence 4 cells.
Local2Stat_col = zeros(nK,1);
Local2P_col = zeros(nK,1);
Local2MeanW_col = zeros(nK,1);
Local2MinW_col = zeros(nK,1);
Local2MaxW_col = zeros(nK,1);
Local2MinCellN_col = zeros(nK,1);
Local2MaxCellN_col = zeros(nK,1);
Local2MinKCell_col = zeros(nK,1);
Local2MaxKCell_col = zeros(nK,1);

% B=3 means 3 bins per covariate, hence 9 cells.
Local3Stat_col = zeros(nK,1);
Local3P_col = zeros(nK,1);
Local3MeanW_col = zeros(nK,1);
Local3MinW_col = zeros(nK,1);
Local3MaxW_col = zeros(nK,1);
Local3MinCellN_col = zeros(nK,1);
Local3MaxCellN_col = zeros(nK,1);
Local3MinKCell_col = zeros(nK,1);
Local3MaxKCell_col = zeros(nK,1);

for ik=1:nK
    k = kList(ik);

    if k < 18
        error('k=%d gives too few tail observations.',k);
    end
    if k >= n
        error('Invalid k=%d.',k);
    end

    qeq = 1-k/n;

    fprintf('============================================\n');
    fprintf('k=%d (k/n=%.5f; equivalent q=%.5f)\n',k,k/n,qeq);
    fprintf('============================================\n');

    %% Global top-k dCov
    [Ydesc,ord] = sort(Y,'descend');
    wG = Ydesc(k+1);
    idxG = ord(1:k);

    XG = X(idxG,:);
    ZG = log(Y(idxG)./wG);

    [TG,pG] = dcov_permutation_test(XG,ZG,Rperm);

    fprintf('Global dCov: statistic=%.8f, p=%.4f, w=%.4f\n',TG,pG,wG);

    %% Local dCov for B=2 and B=3 bins per covariate
    localOut = cell(numel(BList),1);

    for ib=1:numel(BList)
        B = BList(ib);

        [TL,pL,wLocal,cellN,kCell] = ...
            local_balanced_dcov(Y,X,k,B,Rperm);

        localOut{ib} = struct( ...
            'B',B, ...
            'J',B^2, ...
            'stat',TL, ...
            'pvalue',pL, ...
            'wLocal',wLocal, ...
            'cellN',cellN, ...
            'kCell',kCell);

        fprintf('Local %dx%d dCov: statistic=%.8f, p=%.4f\n',B,B,TL,pL);
        fprintf(['  local thresholds: min=%.4f, mean=%.4f, max=%.4f; ' ...
                 'cell n=%d--%d; cell k=%d--%d\n'], ...
            min(wLocal),mean(wLocal),max(wLocal), ...
            min(cellN),max(cellN),min(kCell),max(kCell));
    end
    fprintf('\n');

    %% Store common/global quantities
    K_col(ik)=k;
    Kfrac_col(ik)=k/n;
    EquivalentQ_col(ik)=qeq;

    GlobalW_col(ik)=wG;
    GlobalStat_col(ik)=TG;
    GlobalP_col(ik)=pG;

    %% Store 2x2 local quantities
    out2 = localOut{1};
    Local2Stat_col(ik)=out2.stat;
    Local2P_col(ik)=out2.pvalue;
    Local2MeanW_col(ik)=mean(out2.wLocal);
    Local2MinW_col(ik)=min(out2.wLocal);
    Local2MaxW_col(ik)=max(out2.wLocal);
    Local2MinCellN_col(ik)=min(out2.cellN);
    Local2MaxCellN_col(ik)=max(out2.cellN);
    Local2MinKCell_col(ik)=min(out2.kCell);
    Local2MaxKCell_col(ik)=max(out2.kCell);

    %% Store 3x3 local quantities
    out3 = localOut{2};
    Local3Stat_col(ik)=out3.stat;
    Local3P_col(ik)=out3.pvalue;
    Local3MeanW_col(ik)=mean(out3.wLocal);
    Local3MinW_col(ik)=min(out3.wLocal);
    Local3MaxW_col(ik)=max(out3.wLocal);
    Local3MinCellN_col(ik)=min(out3.cellN);
    Local3MaxCellN_col(ik)=max(out3.cellN);
    Local3MinKCell_col(ik)=min(out3.kCell);
    Local3MaxKCell_col(ik)=max(out3.kCell);
end

%% Results table
results = table( ...
    K_col,Kfrac_col,EquivalentQ_col, ...
    GlobalW_col,GlobalStat_col,GlobalP_col, ...
    Local2Stat_col,Local2P_col, ...
    Local2MeanW_col,Local2MinW_col,Local2MaxW_col, ...
    Local2MinCellN_col,Local2MaxCellN_col, ...
    Local2MinKCell_col,Local2MaxKCell_col, ...
    Local3Stat_col,Local3P_col, ...
    Local3MeanW_col,Local3MinW_col,Local3MaxW_col, ...
    Local3MinCellN_col,Local3MaxCellN_col, ...
    Local3MinKCell_col,Local3MaxKCell_col, ...
    'VariableNames',{ ...
    'k','k_over_n','equivalent_threshold_quantile', ...
    'global_w_hat','global_dcov2','global_pvalue', ...
    'local_2x2_dcov2','local_2x2_pvalue', ...
    'local_2x2_mean_w_hat','local_2x2_min_w_hat','local_2x2_max_w_hat', ...
    'local_2x2_min_cell_n','local_2x2_max_cell_n', ...
    'local_2x2_min_k_per_cell','local_2x2_max_k_per_cell', ...
    'local_3x3_dcov2','local_3x3_pvalue', ...
    'local_3x3_mean_w_hat','local_3x3_min_w_hat','local_3x3_max_w_hat', ...
    'local_3x3_min_cell_n','local_3x3_max_cell_n', ...
    'local_3x3_min_k_per_cell','local_3x3_max_k_per_cell'});

disp(results);

writetable(results,'application_dcov_global_local_J23_results.xlsx','Sheet','Results');
writetable(results,'application_dcov_global_local_J23_results.csv');

settings = table(n,Rperm,level,numWorkers,BList(1),BList(2), ...
    'VariableNames',{ ...
    'usable_positive_claims','permutations','nominal_level', ...
    'parallel_workers','local_bins_small','local_bins_large'});
writetable(settings,'application_dcov_global_local_J23_results.xlsx','Sheet','Settings');

end


function [Tobs,pval,wLocal,cellN,kCell] = ...
    local_balanced_dcov(Y,X,k,B,Rperm)
% LOCAL_BALANCED_DCOV
% B empirical equal-frequency bins for each of the two covariates,
% giving J=B^2 local cells.

bin1 = empirical_equal_frequency_bins(X(:,1),B);
bin2 = empirical_equal_frequency_bins(X(:,2),B);
cellID = (bin1-1)*B + bin2;

J = B^2;
cellN = zeros(J,1);
for j=1:J
    cellN(j) = sum(cellID==j);
end

% Allocate exactly k observations across cells as evenly as possible.
baseK = floor(k/J);
remK = mod(k,J);
kCell = baseK*ones(J,1);

if remK>0
    % Allocate extras to the largest cells. MATLAB sort is deterministic
    % for ties, so this is fully reproducible.
    [~,largeCells] = sort(cellN,'descend');
    kCell(largeCells(1:remK)) = kCell(largeCells(1:remK))+1;
end

if any(cellN<=kCell)
    error('At least one %dx%d local cell is too small for k=%d.',B,B,k);
end

XL = NaN(k,2);
ZL = NaN(k,1);
wLocal = NaN(J,1);

pos = 0;
for j=1:J
    idxCell = find(cellID==j);
    Yj = Y(idxCell);
    Xj = X(idxCell,:);

    kj = kCell(j);
    [YjDesc,ordj] = sort(Yj,'descend');

    wj = YjDesc(kj+1);
    topj = ordj(1:kj);

    sel = (pos+1):(pos+kj);
    XL(sel,:) = Xj(topj,:);
    ZL(sel) = log(Yj(topj)./wj);

    wLocal(j) = wj;
    pos = pos+kj;
end

if pos~=k
    error('Local selection did not produce exactly k observations.');
end

[Tobs,pval] = dcov_permutation_test(XL,ZL,Rperm);
end


function bin = empirical_equal_frequency_bins(x,B)
x = x(:);
n = numel(x);

% Deterministic tie-breaking by observation index.
tmp = [x,(1:n)'];
[~,ord] = sortrows(tmp,[1 2]);

bin = zeros(n,1);
for r=1:n
    b = ceil(B*r/n);
    b = max(1,min(B,b));
    bin(ord(r)) = b;
end
end


function [Tobs,pval] = dcov_permutation_test(X,Z,Rperm)

Z = Z(:);
if isvector(X)
    X = X(:);
end

m = size(X,1);
A = centered_euclidean_distance_matrix(X);
B = centered_euclidean_distance_matrix(Z);

Tobs = sum(A.*B,'all')/(m^2);

Tperm = zeros(Rperm,1);
parfor r=1:Rperm
    pi = randperm(m);
    Bpi = B(pi,pi);
    Tperm(r) = sum(A.*Bpi,'all')/(m^2);
end

pval = (1+sum(Tperm>=Tobs))/(Rperm+1);
end


function A = centered_euclidean_distance_matrix(V)

if isvector(V)
    V = V(:);
end

sqNorm = sum(V.^2,2);
D2 = sqNorm+sqNorm'-2*(V*V');
D2 = max(D2,0);
D = sqrt(D2);

rowMean = mean(D,2);
colMean = mean(D,1);
grandMean = mean(D,'all');

A = D-rowMean-colMean+grandMean;
end
