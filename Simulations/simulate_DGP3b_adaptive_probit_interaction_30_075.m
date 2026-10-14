function results = simulate_DGP3b_adaptive_probit_interaction_30_075_revised()
% SIMULATE_DGP3B_ADAPTIVE_PROBIT_INTERACTION_30_075_REVISED
%
% Original DGP 3b:
%   X1, X2 iid U[-1,1]
%   alpha_theta(X) = 2*exp(theta*X1*X2)
%   kappa(X) = 1
%   Y = U^(-1/alpha_theta(X))
%
% Adaptive localization:
%   - 20% training / 80% inference split
%   - training threshold counts M1 = 30% of training sample
%   - M2 = 25% of M1 = 7.5% of training sample
%   - probit working score uses the generic second-order basis
%         X1, X1^2, X2, X2^2, X1*X2
%   - inference observations are split at the median fitted score
%   - total tail count k is allocated as evenly as possible across
%     the two score groups
%   - cellwise log exceedances are pooled and tested by dCov permutation
%
% Requires Statistics and Machine Learning Toolbox for FITGLM.

    %% Settings
    numWorkers = 8;
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local', numWorkers);
    elseif pool.NumWorkers ~= numWorkers
        delete(pool);
        pool = parpool('local', numWorkers);
    end

    % Keep the same seed used in the earlier DGP 3b interaction experiment
    % so the revised run remains directly comparable.
    baseSeed = 2026100814;

    nList = [2000, 5000];
    kByN  = [100,  50; ...
             200, 100];
    thetaList = [0, 0.5, 1.0, 1.5];

    MC    = 1000;
    Rperm = 299;
    level = 0.05;

    trainFraction = 1/5;
    M1Fraction    = 0.30;
    M2Fraction    = 0.25;   % fraction of M1, hence 7.5% of training sample

    fprintf('DGP 3b: adaptive dCov with quadratic-interaction probit\n');
    fprintf('Score basis: X1, X1^2, X2, X2^2, X1*X2\n');
    fprintf('Training split = 20%%; M1 = 30%% of training; M2 = 7.5%% of training\n');
    fprintf('MC = %d; permutations = %d\n\n', MC, Rperm);

    %% Storage
    nDesigns = numel(nList) * size(kByN,2) * numel(thetaList);

    N_col            = zeros(nDesigns,1);
    K_col            = zeros(nDesigns,1);
    Theta_col        = zeros(nDesigns,1);
    AlphaMin_col     = zeros(nDesigns,1);
    AlphaMax_col     = zeros(nDesigns,1);
    TrainN_col       = zeros(nDesigns,1);
    InferN_col       = zeros(nDesigns,1);
    M1_col           = zeros(nDesigns,1);
    M2_col           = zeros(nDesigns,1);
    Reject_col       = NaN(nDesigns,1);
    MCSE_col         = NaN(nDesigns,1);
    Fail_col         = zeros(nDesigns,1);
    MeanW1_col       = NaN(nDesigns,1);
    MeanW2_col       = NaN(nDesigns,1);
    MeanDev_col      = NaN(nDesigns,1);
    MeanBetaNorm_col = NaN(nDesigns,1);
    MeanScoreSd_col  = NaN(nDesigns,1);
    MeanAvgW_col     = NaN(nDesigns,1);
    MeanMinW_col     = NaN(nDesigns,1);
    MeanMaxW_col     = NaN(nDesigns,1);
    Perm_col         = Rperm * ones(nDesigns,1);
    Level_col        = level * ones(nDesigns,1);

    replications = cell(nDesigns,1);
    row = 0;
    tic;

    %% Monte Carlo
    for in = 1:numel(nList)
        n = nList(in);

        for ik = 1:size(kByN,2)
            k = kByN(in,ik);

            for it = 1:numel(thetaList)
                theta = thetaList(it);
                row = row + 1;
                designID = row;

                rejected   = false(MC,1);
                pvalues    = NaN(MC,1);
                failed     = false(MC,1);
                w1Rep      = NaN(MC,1);
                w2Rep      = NaN(MC,1);
                devRep     = NaN(MC,1);
                betaRep    = NaN(MC,6);  % intercept + five score terms
                scoreSdRep = NaN(MC,1);
                localWRep  = NaN(MC,2);

                fprintf('Running n=%d, k=%d, theta=%.2f ...\n', n, k, theta);

                parfor rep = 1:MC
                    rng(baseSeed + 100000*designID + rep, 'twister');

                    % --------------------------------------------------
                    % Generate original DGP 3b
                    % --------------------------------------------------
                    X1 = 2*rand(n,1) - 1;
                    X2 = 2*rand(n,1) - 1;
                    U  = rand(n,1);
                    U(U == 0) = realmin;

                    alpha = 2 .* exp(theta .* X1 .* X2);
                    Y = U .^ (-1 ./ alpha);
                    X = [X1, X2];

                    try
                        [pval, info] = adaptive_test_one_rep( ...
                            X, Y, k, Rperm, trainFraction, M1Fraction, M2Fraction);

                        pvalues(rep)    = pval;
                        rejected(rep)   = (pval <= level);
                        w1Rep(rep)      = info.training_w1;
                        w2Rep(rep)      = info.training_w2;
                        devRep(rep)     = info.probit_deviance;
                        betaRep(rep,:)  = info.probit_beta.';
                        scoreSdRep(rep) = info.score_sd;
                        localWRep(rep,:)= info.local_w.';
                    catch
                        failed(rep) = true;
                    end
                end

                used = ~failed & isfinite(pvalues);
                nUsed = sum(used);

                mTrain = floor(trainFraction*n);
                M1 = floor(M1Fraction*mTrain);
                M2 = floor(M2Fraction*M1);

                N_col(row)        = n;
                K_col(row)        = k;
                Theta_col(row)    = theta;
                AlphaMin_col(row) = 2*exp(-abs(theta));
                AlphaMax_col(row) = 2*exp(abs(theta));
                TrainN_col(row)   = mTrain;
                InferN_col(row)   = n - mTrain;
                M1_col(row)       = M1;
                M2_col(row)       = M2;
                Fail_col(row)     = MC - nUsed;

                if nUsed > 0
                    rr = mean(rejected(used));
                    Reject_col(row)       = rr;
                    MCSE_col(row)         = sqrt(rr*(1-rr)/nUsed);
                    MeanW1_col(row)       = mean(w1Rep(used));
                    MeanW2_col(row)       = mean(w2Rep(used));
                    MeanDev_col(row)      = mean(devRep(used));
                    MeanBetaNorm_col(row) = mean(sqrt(sum(betaRep(used,:).^2,2)));
                    MeanScoreSd_col(row)  = mean(scoreSdRep(used));
                    MeanAvgW_col(row)     = mean(mean(localWRep(used,:),2));
                    MeanMinW_col(row)     = mean(min(localWRep(used,:),[],2));
                    MeanMaxW_col(row)     = mean(max(localWRep(used,:),[],2));
                end

                replications{row} = struct( ...
                    'n', n, 'k', k, 'theta', theta, ...
                    'pvalue', pvalues, 'reject', rejected, 'failed', failed, ...
                    'training_w1', w1Rep, 'training_w2', w2Rep, ...
                    'probit_deviance', devRep, 'probit_beta', betaRep, ...
                    'score_sd', scoreSdRep, 'local_w_hat', localWRep);

                fprintf('  rejection rate = %.4f, MCSE = %.4f, failures = %d/%d\n', ...
                    Reject_col(row), MCSE_col(row), Fail_col(row), MC);
            end
        end
    end

    toc;

    %% Results table
    results = table( ...
        N_col, K_col, Theta_col, AlphaMin_col, AlphaMax_col, ...
        TrainN_col, InferN_col, M1_col, M2_col, Reject_col, MCSE_col, Fail_col, ...
        MeanW1_col, MeanW2_col, MeanDev_col, MeanBetaNorm_col, MeanScoreSd_col, ...
        MeanAvgW_col, MeanMinW_col, MeanMaxW_col, Perm_col, Level_col, ...
        'VariableNames', { ...
        'n','k','theta','alpha_min','alpha_max','training_n','inference_n', ...
        'M1','M2','rejection_rate','mcse','failures', ...
        'mean_training_w1','mean_training_w2','mean_probit_deviance', ...
        'mean_beta_norm','mean_score_sd','mean_average_local_w_hat', ...
        'mean_min_local_w_hat','mean_max_local_w_hat', ...
        'permutations','nominal_level'});

    disp(results);

    outBase = 'DGP3b_dcov_adaptive_probit_interaction_30_075_revised';
    writetable(results, [outBase '_summary.xlsx'], 'Sheet', 'Summary');
    writetable(results, [outBase '_results.csv']);

    settings = table( ...
        MC, Rperm, level, numWorkers, baseSeed, ...
        trainFraction, M1Fraction, M2Fraction, ...
        'VariableNames', { ...
        'MC_replications','permutations','nominal_level','parallel_workers', ...
        'base_seed','training_fraction','M1_fraction_of_training','M2_fraction_of_M1'});

    writetable(settings, [outBase '_summary.xlsx'], 'Sheet', 'Settings');

    save([outBase '_replications.mat'], ...
        'replications','results','settings','nList','kByN','thetaList', ...
        'MC','Rperm','level','trainFraction','M1Fraction','M2Fraction', ...
        'numWorkers','baseSeed','-v7.3');

    %% Power plot
    figure;
    hold on;
    for in = 1:numel(nList)
        for ik = 1:size(kByN,2)
            n = nList(in);
            k = kByN(in,ik);
            rows = (results.n == n) & (results.k == k);
            plot(results.theta(rows), results.rejection_rate(rows), '-o', ...
                'LineWidth', 1.3, ...
                'DisplayName', sprintf('n=%d, k=%d', n, k));
        end
    end
    yline(level, '--', 'Nominal 5%', 'LineWidth', 1.1);
    xlabel('\theta');
    ylabel('Rejection probability');
    title('DGP 3b: adaptive quadratic-interaction probit dCov');
    legend('Location','best');
    grid on;
    hold off;
end


function [pval, info] = adaptive_test_one_rep(X, Y, k, Rperm, trainFraction, M1Fraction, M2Fraction)
% One adaptive-locally-balanced dCov replication.

    n = size(X,1);
    d = size(X,2);

    % --------------------------------------------------------------
    % 1. Random training/inference split
    % --------------------------------------------------------------
    mTrain = floor(trainFraction*n);
    splitOrder = randperm(n);
    idxTrain = splitOrder(1:mTrain);
    idxInf   = splitOrder((mTrain+1):end);

    Xtr  = X(idxTrain,:);
    Ytr  = Y(idxTrain);
    Xinf = X(idxInf,:);
    Yinf = Y(idxInf);
    nInf = numel(idxInf);

    % --------------------------------------------------------------
    % 2. Two training thresholds
    % --------------------------------------------------------------
    M1 = floor(M1Fraction*mTrain);
    M2 = floor(M2Fraction*M1);

    if M1 < 10 || M2 < 1 || M2 >= M1 || M1 >= mTrain
        error('Invalid training-tail counts.');
    end

    [YtrDesc, orderTr] = sort(Ytr, 'descend');
    w1 = YtrDesc(M1+1);
    w2 = YtrDesc(M2+1);

    fitIdx = orderTr(1:M1);
    Xfit = Xtr(fitIdx,:);
    D = double(Ytr(fitIdx) > w2);

    % --------------------------------------------------------------
    % 3. Probit ranking model: generic second-order basis
    %    For d=2: X1, X1^2, X2, X2^2, X1*X2
    % --------------------------------------------------------------
    Qfit = second_order_basis(Xfit);
    Qinf = second_order_basis(Xinf);

    mdl = fitglm(Qfit, D, 'linear', ...
        'Distribution', 'binomial', 'Link', 'probit');

    pScore = predict(mdl, Qinf);
    pScore = pScore(:);

    if any(~isfinite(pScore))
        error('Nonfinite probit scores.');
    end

    % --------------------------------------------------------------
    % 4. Median score split on inference sample
    % --------------------------------------------------------------
    tieMatrix = [pScore, Xinf, (1:nInf)'];
    [~, scoreOrder] = sortrows(tieMatrix, 1:size(tieMatrix,2));

    nLow = floor(nInf/2);
    cellID = 2*ones(nInf,1);
    cellID(scoreOrder(1:nLow)) = 1;

    kCell = balanced_counts(k,2).';
    cellN = [sum(cellID==1); sum(cellID==2)];

    if any(cellN <= kCell)
        error('An inference cell has too few observations.');
    end

    % --------------------------------------------------------------
    % 5. Cellwise tail selection
    % --------------------------------------------------------------
    Xe = NaN(k,d);
    Ze = NaN(k,1);
    wLocal = NaN(2,1);

    pos = 0;
    for j = 1:2
        idxCell = find(cellID == j);
        Xj = Xinf(idxCell,:);
        Yj = Yinf(idxCell);
        kj = kCell(j);

        [YjDesc, ord] = sort(Yj, 'descend');
        wj = YjDesc(kj+1);
        top = ord(1:kj);

        sel = (pos+1):(pos+kj);
        Xe(sel,:) = Xj(top,:);
        Ze(sel)   = log(Yj(top)./wj);

        wLocal(j) = wj;
        pos = pos + kj;
    end

    % --------------------------------------------------------------
    % 6. dCov permutation test
    % --------------------------------------------------------------
    pval = dcov_permutation_pvalue(Xe, Ze, Rperm);

    info = struct();
    info.training_w1 = w1;
    info.training_w2 = w2;
    info.probit_beta = mdl.Coefficients.Estimate;
    info.probit_deviance = mdl.Deviance;
    info.score_sd = std(pScore);
    info.local_w = wLocal;
end


function Q = second_order_basis(X)
% Linear terms, squares, and all pairwise interactions; no intercept.
% FITGLM supplies the intercept automatically.

    [n,d] = size(X);
    nInt = d*(d-1)/2;
    Q = zeros(n, 2*d + nInt);

    col = 0;
    for j = 1:d
        col = col + 1;
        Q(:,col) = X(:,j);
        col = col + 1;
        Q(:,col) = X(:,j).^2;
    end

    for j = 1:(d-1)
        for ell = (j+1):d
            col = col + 1;
            Q(:,col) = X(:,j).*X(:,ell);
        end
    end
end


function counts = balanced_counts(total, J)
% Split an integer total as evenly as possible across J groups.

    counts = floor(total/J)*ones(1,J);
    r = mod(total,J);
    if r > 0
        counts(1:r) = counts(1:r) + 1;
    end
end


function pval = dcov_permutation_pvalue(X, Z, Rperm)
% Distance-covariance permutation p-value using the biased sample dCov^2.

    n = size(X,1);
    if numel(Z) ~= n
        error('X and Z must have the same number of rows.');
    end

    A = pairwise_distance_matrix(X);
    B = pairwise_distance_matrix(Z(:));

    A = double_center(A);
    B = double_center(B);

    statObs = mean(A(:).*B(:));

    countGE = 0;
    for b = 1:Rperm
        perm = randperm(n);
        Bp = B(perm,perm);
        statPerm = mean(A(:).*Bp(:));
        if statPerm >= statObs
            countGE = countGE + 1;
        end
    end

    pval = (1 + countGE)/(Rperm + 1);
end


function D = pairwise_distance_matrix(X)
% Euclidean pairwise distance matrix.

    G = sum(X.^2,2);
    D2 = max(G + G.' - 2*(X*X.'), 0);
    D = sqrt(D2);
end


function A = double_center(D)
% Double-center a distance matrix.

    rowMean = mean(D,2);
    colMean = mean(D,1);
    grandMean = mean(D(:));
    A = D - rowMean - colMean + grandMean;
end
