function [pval, info] = adaptive_balanced_dcov_probit_second_order(X, Y, k, cfg)
% ADAPTIVE_BALANCED_DCOV_PROBIT_SECOND_ORDER
% Sample-split adaptive locally balanced dCov test.
%
% The training score uses a generic second-order probit basis:
%   - all linear terms X_j
%   - all squares X_j^2
%   - all pairwise interactions X_j X_l, j<l
%
% For d=1 this reduces to (X, X^2).
% For d=2 this is (X1, X1^2, X2, X2^2, X1*X2).
%
% cfg fields:
%   trainFraction : training fraction of full sample
%   M1Fraction    : M1 as a fraction of training sample
%   M2Fraction    : M2 as a fraction of M1
%   Rperm         : number of random permutations

    n = size(X,1);
    d = size(X,2);
    Y = Y(:);

    if numel(Y) ~= n
        error('X and Y must have the same number of observations.');
    end
    if ~all(isfield(cfg, {'trainFraction','M1Fraction','M2Fraction','Rperm'}))
        error('cfg is missing one or more required fields.');
    end

    % --------------------------------------------------------------
    % 1. Random training/inference split
    % --------------------------------------------------------------
    mTrain = floor(cfg.trainFraction*n);
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
    M1 = floor(cfg.M1Fraction*mTrain);
    M2 = floor(cfg.M2Fraction*M1);

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
    % 3. Probit ranking model with generic second-order basis
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
    % 5. Cellwise tail selection and local log exceedances
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
    pval = dcov_permutation_pvalue(Xe, Ze, cfg.Rperm);

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
% Distance-covariance permutation p-value using biased sample dCov^2.

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
