function results = simulate_DGP1_beta1_local_J23()
% SIMULATE_DGP1_BETA1_LOCAL_J23
% DGP 1: Hall-type null with LOCALLY BALANCED dCov sampling.
%
% DGP 1 (beta = 1 version):
%   X ~ Unif[-1,1]
%   s(X) = exp(X/2)
%   Y = s(X)*(U^(-1/2)-1), U ~ Unif(0,1)
%
% Hence
%   P(Y > y | X=x)
%      = [1 + y/s(x)]^(-2)
%      = s(x)^2*y^(-2)*{1 - 2*s(x)*y^(-1)
%                        + 3*s(x)^2*y^(-2) + O(y^(-3))}.
%
% Thus alpha0 = 2 and the Hall second-order exponent is beta = 1.
% The leading tail-frequency term is kappa(x)=s(x)^2=exp(x),
% while the second-order coefficient is D(x)=-2*s(x)=-2*exp(x/2).
%
% LOCAL-WINDOW PROCEDURE:
%   1. Run the locally balanced procedure with J=2 and J=3.
%   2. Divide X into J empirical-quantile windows by sorting X and
%      taking consecutive blocks. When n is not divisible by J, the
%      window sizes differ by at most one observation.
%   3. Allocate the total tail size k across the J windows as evenly
%      as possible. When k is not divisible by J, the local tail sizes
%      differ by at most one and still sum exactly to k.
%   4. In window j, let w_j be the next local order statistic after
%      the k_j selected observations:
%
%         w_j = Y_{(n_j-k_j):n_j}.
%
%   5. For each selected observation form
%
%         Z_i^* = log(Y_i / w_j).
%
%   6. Pool all selected observations across the J windows. The pooled
%      sample contains exactly k observations.
%   7. Compute dCov between pooled X and pooled Z^*, and calibrate
%      using GLOBAL permutation of the pooled Z^* values.
%
% Tail sizes:
%   n=2000: k=100,50
%   n=5000: k=200,100
%
% Settings:
%   J     = 2,3
%   MC    = 1000
%   Rperm = 299
%   level = 0.05
%
% Outputs:
%   DGP1_beta1_dcov_local_J23_summary.xlsx
%   DGP1_beta1_dcov_local_J23_results.csv
%   DGP1_beta1_dcov_local_J23_replications.mat

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

nList = [2000,5000];

kByN = [100, 50; ...
        200,100];

JList = [2,3];
MC    = 1000;
Rperm = 299;
level = 0.05;

fprintf('DGP 1: Hall-type null, locally balanced dCov\n');
fprintf('alpha0=2, beta=1, J in {%s}\n', num2str(JList));
fprintf('MC=%d, permutations=%d\n\n',MC,Rperm);

%% Storage

nDesigns = numel(JList)*numel(nList)*size(kByN,2);

N_col           = zeros(nDesigns,1);
K_col           = zeros(nDesigns,1);
J_col           = zeros(nDesigns,1);
NperWinMin_col  = zeros(nDesigns,1);
NperWinMax_col  = zeros(nDesigns,1);
KperWinMin_col  = zeros(nDesigns,1);
KperWinMax_col  = zeros(nDesigns,1);
Kfrac_col       = zeros(nDesigns,1);

Reject_col      = zeros(nDesigns,1);
MCSE_col        = zeros(nDesigns,1);

AvgW_col        = zeros(nDesigns,1);
SdAvgW_col      = zeros(nDesigns,1);
MinW_col        = zeros(nDesigns,1);
MaxW_col        = zeros(nDesigns,1);

Perm_col        = Rperm*ones(nDesigns,1);
Level_col       = level*ones(nDesigns,1);

replications = cell(nDesigns,1);

row = 0;
tic;

for iJ = 1:numel(JList)

    J = JList(iJ);

    for in = 1:numel(nList)

        n = nList(in);

        % Empirical-quantile window boundaries. This gives J blocks whose
        % sizes differ by at most one, and whose union is exactly 1:n.
        nEdges = round(linspace(0,n,J+1));
        nPerWindow = diff(nEdges);

        for ik = 1:size(kByN,2)

            k = kByN(in,ik);

            % Allocate k as evenly as possible across the J windows while
            % preserving the total pooled tail size exactly.
            kEdges = round(linspace(0,k,J+1));
            kPerWindow = diff(kEdges);

            if any(kPerWindow < 1)
                error('Each local window must contribute at least one tail observation.');
            end

            if any(kPerWindow + 1 > nPerWindow)
                error('A local tail size is too large for its window.');
            end

            designID = ((iJ-1)*numel(nList) + (in-1))*size(kByN,2) + ik;

            rejected      = false(MC,1);
            pvalues       = NaN(MC,1);

            avgWHatRep    = NaN(MC,1);
            minWHatRep    = NaN(MC,1);
            maxWHatRep    = NaN(MC,1);
            localWHatRep  = NaN(MC,J);

            fprintf(['Running J=%d, n=%d, k=%d, ' ...
                     'local k range=[%d,%d] ...\n'], ...
                J,n,k,min(kPerWindow),max(kPerWindow));

            parfor rep = 1:MC

                rng(baseSeed + 100000*designID + rep,'twister');

                % ---------------------------------------------------------
                % Generate DGP 1: beta = 1
                % ---------------------------------------------------------
                X = 2*rand(n,1)-1;
                U = rand(n,1);
                U(U==0) = realmin;

                s = exp(X/2);
                Y = s.*(U.^(-1/2)-1);

                % ---------------------------------------------------------
                % Divide X into J empirical-quantile windows.
                % Sorting X and taking consecutive blocks gives nearly
                % equal window sizes when n is not divisible by J.
                % ---------------------------------------------------------
                [~,xOrder] = sort(X,'ascend');

                Xe = NaN(k,1);
                Ze = NaN(k,1);
                wLocal = NaN(J,1);

                pos = 0;

                for j = 1:J

                    firstIdx = nEdges(j) + 1;
                    lastIdx  = nEdges(j+1);

                    winIdx = xOrder(firstIdx:lastIdx);

                    Xj = X(winIdx);
                    Yj = Y(winIdx);

                    kj = kPerWindow(j);

                    % Local threshold: the next order statistic after the
                    % kj selected observations in this window.
                    [YjDesc,orderLocal] = sort(Yj,'descend');

                    wj = YjDesc(kj+1);
                    topLocal = orderLocal(1:kj);

                    sel = (pos+1):(pos+kj);

                    Xe(sel) = Xj(topLocal);
                    Ze(sel) = log(Yj(topLocal)./wj);

                    wLocal(j) = wj;
                    pos = pos+kj;
                end

                if pos ~= k
                    error('Local selection did not produce exactly k observations.');
                end

                % ---------------------------------------------------------
                % Global permutation dCov test on pooled local exceedances
                % ---------------------------------------------------------
                pval = dcov_permutation_pvalue(Xe,Ze,Rperm);

                pvalues(rep)  = pval;
                rejected(rep) = (pval<=level);

                localWHatRep(rep,:) = wLocal.';
                avgWHatRep(rep) = mean(wLocal);
                minWHatRep(rep) = min(wLocal);
                maxWHatRep(rep) = max(wLocal);
            end

            row = row+1;

            rejectRate = mean(rejected);

            N_col(row)          = n;
            K_col(row)          = k;
            J_col(row)          = J;
            NperWinMin_col(row) = min(nPerWindow);
            NperWinMax_col(row) = max(nPerWindow);
            KperWinMin_col(row) = min(kPerWindow);
            KperWinMax_col(row) = max(kPerWindow);
            Kfrac_col(row)      = k/n;

            Reject_col(row)     = rejectRate;
            MCSE_col(row)       = sqrt(rejectRate*(1-rejectRate)/MC);

            AvgW_col(row)       = mean(avgWHatRep);
            SdAvgW_col(row)     = std(avgWHatRep);
            MinW_col(row)       = mean(minWHatRep);
            MaxW_col(row)       = mean(maxWHatRep);

            replications{row} = struct( ...
                'n',n, ...
                'k',k, ...
                'J',J, ...
                'n_per_window',nPerWindow, ...
                'k_per_window',kPerWindow, ...
                'k_over_n',k/n, ...
                'pvalue',pvalues, ...
                'reject',rejected, ...
                'local_w_hat',localWHatRep, ...
                'average_local_w_hat',avgWHatRep, ...
                'minimum_local_w_hat',minWHatRep, ...
                'maximum_local_w_hat',maxWHatRep);

            fprintf(['  rejection rate=%.4f, MCSE=%.4f, ' ...
                     'mean average local threshold=%.4f\n'], ...
                     Reject_col(row),MCSE_col(row),AvgW_col(row));
        end
    end
end

toc;

%% Save

results = table( ...
    N_col,K_col,J_col,NperWinMin_col,NperWinMax_col, ...
    KperWinMin_col,KperWinMax_col,Kfrac_col, ...
    Reject_col,MCSE_col, ...
    AvgW_col,SdAvgW_col,MinW_col,MaxW_col, ...
    Perm_col,Level_col, ...
    'VariableNames', { ...
        'n','k','windows','n_per_window_min','n_per_window_max', ...
        'k_per_window_min','k_per_window_max','k_over_n', ...
        'rejection_rate','mcse', ...
        'mean_average_local_w_hat','sd_average_local_w_hat', ...
        'mean_min_local_w_hat','mean_max_local_w_hat', ...
        'permutations','nominal_level'});

disp(results);

writetable(results, ...
    'DGP1_beta1_dcov_local_J23_summary.xlsx', ...
    'Sheet','Summary');

writetable(results, ...
    'DGP1_beta1_dcov_local_J23_results.csv');

J_values = {strtrim(sprintf('%d ',JList))};
settings = table( ...
    MC,Rperm,level,numWorkers,baseSeed,J_values, ...
    'VariableNames', { ...
        'MC_replications','permutations','nominal_level', ...
        'parallel_workers','base_seed','J_values'});

writetable(settings, ...
    'DGP1_beta1_dcov_local_J23_summary.xlsx', ...
    'Sheet','Settings');

save('DGP1_beta1_dcov_local_J23_replications.mat', ...
    'replications','results','settings', ...
    'nList','kByN','JList','MC','Rperm','level', ...
    'numWorkers','baseSeed','-v7.3');

%% Plot empirical size

figure;
hold on;

for iJ = 1:numel(JList)
    J = JList(iJ);

    for in = 1:numel(nList)

        rows = (results.n==nList(in)) & (results.windows==J);

        [xplot,ord] = sort(results.k_over_n(rows));
        yplot = results.rejection_rate(rows);
        yplot = yplot(ord);

        plot(xplot,yplot,'-o', ...
            'LineWidth',1.4, ...
            'DisplayName',sprintf('n=%d, J=%d',nList(in),J));
    end
end

yline(level,'--','Nominal 5%','LineWidth',1.2);

xlabel('Tail fraction k/n');
ylabel('Rejection probability');
title('DGP 1: locally balanced dCov size, beta=1, J=2 and 3');
legend('Location','best');
grid on;
hold off;

end


function pval = dcov_permutation_pvalue(X,Z,Rperm)
% Global permutation p-value for empirical squared distance covariance.

X = X(:);
Z = Z(:);

m = numel(X);

if numel(Z) ~= m
    error('X and Z must have the same number of observations.');
end

A = centered_distance_matrix(X);
B = centered_distance_matrix(Z);

Tobs = sum(A.*B,'all')/(m^2);

nGreaterEqual = 0;

for r = 1:Rperm

    pi = randperm(m);
    Bpi = B(pi,pi);

    Tperm = sum(A.*Bpi,'all')/(m^2);

    nGreaterEqual = nGreaterEqual+(Tperm>=Tobs);
end

pval = (1+nGreaterEqual)/(Rperm+1);

end


function A = centered_distance_matrix(V)

V = V(:);

D = abs(V-V.');

rowMean = mean(D,2);
colMean = mean(D,1);
grandMean = mean(D,'all');

A = D-rowMean-colMean+grandMean;

end
