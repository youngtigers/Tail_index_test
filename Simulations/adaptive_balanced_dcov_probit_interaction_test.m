function [pval, info] = adaptive_balanced_dcov_probit_interaction_test(X, Y, k, cfg)
% ADAPTIVE_BALANCED_DCOV_PROBIT_INTERACTION_TEST
% One replication of the two-threshold, training/inference-split,
% adaptively balanced dCov test using a quadratic probit score with pairwise interactions.
%
% INPUTS
%   X   : n-by-d covariate matrix (a vector is treated as one covariate)
%   Y   : n-by-1 positive outcome vector
%   k   : exact total number of inference-sample tail observations retained
%   cfg : structure with fields
%         .trainFraction  (default 1/5)
%         .M1Fraction     (default 1/4 of training size)
%         .M2Fraction     (default 1/2 of M1)
%         .Rperm          number of random permutations (default 299)
%
% PROCEDURE
%   1. Randomly split the sample into training and inference subsamples.
%   2. In the training sample, choose two upper order-statistic thresholds
%      w1 < w2. Among the M1 observations above w1, define D=1{Y>w2}.
%      Under continuity, exactly M2 labels equal one.
%   3. Fit the probit model
%
%        Pr(D=1|X,Y>w1) = Phi(b0 + quadratic main effects + pairwise interactions).
%
%      For the bivariate DGP 3b design, the regressors are
%      (X1,X1^2,X2,X2^2,X1*X2).
%   4. Predict the probit probability on the inference sample, rank the
%      observations by this score, and split them into two equal-size cells.
%      Score ties are broken deterministically by X and then by index.
%   5. Retain approximately k/2 upper Y observations within each cell,
%      normalize by the cell-specific next order statistic, pool, and use a
%      global dCov permutation test.
%
% Requires Statistics and Machine Learning Toolbox for FITGLM.

    if nargin < 4
        error('adaptive_balanced_dcov_probit_interaction_test requires X, Y, k, and cfg.');
    end

    if isvector(X)
        X = X(:);
    end
    Y = Y(:);

    n = size(X,1);
    d = size(X,2);

    if numel(Y) ~= n
        error('X and Y must contain the same number of observations.');
    end
    if any(~isfinite(X(:))) || any(~isfinite(Y)) || any(Y <= 0)
        error('X and Y must be finite, and Y must be positive.');
    end
    if ~(isscalar(k) && k == floor(k) && k >= 2)
        error('k must be an integer of at least 2.');
    end

    cfg = fill_defaults(cfg);

    % ------------------------------------------------------------------
    % Random training/inference split
    % ------------------------------------------------------------------
    mTrain = floor(cfg.trainFraction*n);
    if mTrain < 12
        error('The training subsample is too small.');
    end

    splitOrder = randperm(n);
    idxTrain = splitOrder(1:mTrain);
    idxInf   = splitOrder((mTrain+1):end);

    Xtr = X(idxTrain,:);
    Ytr = Y(idxTrain);
    Xinf = X(idxInf,:);
    Yinf = Y(idxInf);

    nInf = numel(idxInf);

    % ------------------------------------------------------------------
    % Two training thresholds and balanced binary response
    % ------------------------------------------------------------------
    M1 = floor(cfg.M1Fraction*mTrain);
    M2 = floor(cfg.M2Fraction*M1);

    if M1 < (2*d + 6) || M2 < 1 || M2 >= M1 || M1 >= mTrain
        error('Invalid or too-small training-tail counts M1 and M2.');
    end

    [YtrDesc, orderTr] = sort(Ytr, 'descend');
    w1 = YtrDesc(M1 + 1);
    w2 = YtrDesc(M2 + 1);

    trainExcLocal = orderTr(1:M1);
    Xfit = Xtr(trainExcLocal,:);
    D = double(Ytr(trainExcLocal) > w2);

    if sum(D == 1) ~= M2 || sum(D == 0) ~= (M1 - M2)
        error('Training labels do not have the intended counts; check tail ties.');
    end

    % Quadratic basis with pairwise interactions. For d=2 this is
    % (X1,X1^2,X2,X2^2,X1*X2).
    Qfit = quadratic_interaction_basis(Xfit);
    Qinf = quadratic_interaction_basis(Xinf);

    % A working probit classifier. Correct specification is not required
    % for null validity because the score is learned on the independent
    % training subsample.
    mdl = fitglm(Qfit, D, 'linear', ...
        'Distribution', 'binomial', ...
        'Link', 'probit');

    pScore = predict(mdl, Qinf);
    pScore = pScore(:);

    if numel(pScore) ~= nInf || any(~isfinite(pScore))
        error('The fitted probit score has the wrong size or contains nonfinite values.');
    end

    % ------------------------------------------------------------------
    % Equal-size score-rank cells on the inference subsample
    % ------------------------------------------------------------------
    % Lexicographic tie-breaking uses X first and the fixed inference index
    % only if both the score and covariates tie exactly.
    tieMatrix = [pScore, Xinf, (1:nInf)'];
    [~, scoreOrder] = sortrows(tieMatrix, 1:size(tieMatrix,2));

    nCellLow = floor(nInf/2);
    cellID = 2*ones(nInf,1);
    cellID(scoreOrder(1:nCellLow)) = 1;

    cellN = [sum(cellID == 1); sum(cellID == 2)];
    kCell = balanced_counts(k, 2).';

    if any(cellN <= kCell)
        error('An inference cell has too few observations for its local threshold.');
    end

    % ------------------------------------------------------------------
    % Cellwise tail selection and pooled log exceedances
    % ------------------------------------------------------------------
    Xe = NaN(k,d);
    Ze = NaN(k,1);
    wLocal = NaN(2,1);

    pos = 0;
    for j = 1:2
        idxCell = find(cellID == j);
        Xj = Xinf(idxCell,:);
        Yj = Yinf(idxCell);
        kj = kCell(j);

        [YjDesc, orderLocal] = sort(Yj, 'descend');
        wj = YjDesc(kj + 1);
        localTop = orderLocal(1:kj);

        sel = (pos+1):(pos+kj);
        Xe(sel,:) = Xj(localTop,:);
        Ze(sel) = log(Yj(localTop)./wj);

        wLocal(j) = wj;
        pos = pos + kj;
    end

    if pos ~= k || any(~isfinite(Xe(:))) || any(~isfinite(Ze))
        error('Adaptive local selection failed to produce k finite pairs.');
    end

    pval = dcov_permutation_pvalue(Xe, Ze, cfg.Rperm);

    % ------------------------------------------------------------------
    % Compact diagnostics
    % ------------------------------------------------------------------
    info = struct();
    info.training_n = mTrain;
    info.inference_n = nInf;
    info.M1 = M1;
    info.M2 = M2;
    info.training_w1 = w1;
    info.training_w2 = w2;
    info.probit_beta = mdl.Coefficients.Estimate;
    info.probit_deviance = mdl.Deviance;
    info.num_basis_terms = size(Qfit,2);
    info.score_mean = mean(pScore);
    info.score_sd = std(pScore);
    info.score_min = min(pScore);
    info.score_max = max(pScore);
    info.cell_n = cellN;
    info.cell_k = kCell;
    info.local_w = wLocal;
end


function cfg = fill_defaults(cfg)
    defaults = struct( ...
        'trainFraction', 1/5, ...
        'M1Fraction', 1/4, ...
        'M2Fraction', 1/2, ...
        'Rperm', 299);

    names = fieldnames(defaults);
    for j = 1:numel(names)
        name = names{j};
        if ~isfield(cfg, name) || isempty(cfg.(name))
            cfg.(name) = defaults.(name);
        end
    end

    if ~(cfg.trainFraction > 0 && cfg.trainFraction < 1)
        error('cfg.trainFraction must lie in (0,1).');
    end
    if ~(cfg.M1Fraction > 0 && cfg.M1Fraction < 1)
        error('cfg.M1Fraction must lie in (0,1).');
    end
    if ~(cfg.M2Fraction > 0 && cfg.M2Fraction < 1)
        error('cfg.M2Fraction must lie in (0,1).');
    end
    if ~(cfg.Rperm >= 1 && cfg.Rperm == floor(cfg.Rperm))
        error('cfg.Rperm must be a positive integer.');
    end
end


function Q = quadratic_interaction_basis(X)
% Return linear terms, squares, and all pairwise interactions, without an intercept.
% For d=2: (X1,X1^2,X2,X2^2,X1*X2).

    if isvector(X)
        X = X(:);
    end

    [n,d] = size(X);
    nInt = d*(d-1)/2;
    Q = zeros(n,2*d+nInt);

    for j = 1:d
        Q(:,2*j-1) = X(:,j);
        Q(:,2*j)   = X(:,j).^2;
    end

    col = 2*d;
    for j = 1:(d-1)
        for ell = (j+1):d
            col = col + 1;
            Q(:,col) = X(:,j).*X(:,ell);
        end
    end
end

function counts = balanced_counts(total, J)
% Split TOTAL into J integer counts differing by at most one.

    base = floor(total/J);
    r = mod(total,J);
    counts = base*ones(1,J);
    if r > 0
        counts(1:r) = counts(1:r) + 1;
    end
end


function pval = dcov_permutation_pvalue(X, Z, Rperm)
% Global permutation calibration after pooling the two adaptive cells.

    if isvector(X)
        X = X(:);
    end
    Z = Z(:);

    m = size(X,1);
    if numel(Z) ~= m
        error('X and Z must have the same number of observations.');
    end

    A = centered_euclidean_distance_matrix(X);
    B = centered_euclidean_distance_matrix(Z);

    Tobs = sum(A.*B, 'all')/(m^2);

    nGreaterEqual = 0;
    for r = 1:Rperm
        pi = randperm(m);
        Bpi = B(pi,pi);
        Tperm = sum(A.*Bpi, 'all')/(m^2);
        nGreaterEqual = nGreaterEqual + (Tperm >= Tobs);
    end

    pval = (1 + nGreaterEqual)/(Rperm + 1);
end


function A = centered_euclidean_distance_matrix(V)
    if isvector(V)
        V = V(:);
    end

    sqNorm = sum(V.^2,2);
    D2 = sqNorm + sqNorm.' - 2*(V*V.');
    D2 = max(D2,0);
    D = sqrt(D2);

    rowMean = mean(D,2);
    colMean = mean(D,1);
    grandMean = mean(D,'all');

    A = D - rowMean - colMean + grandMean;
end
