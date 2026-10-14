function results = simulate_DGP1_adaptive_probit_30_075()
% SIMULATE_DGP1_ADAPTIVE_PROBIT_30_075
% DGP 1: size of the adaptive, locally balanced dCov test using a
% two-threshold quadratic probit score.
%
% DGP:
%   X ~ Unif[-1,1]
%   s(X) = exp(X/2)
%   Y = s(X)*(U^(-1/2)-1), U ~ Unif(0,1)
%
% The tail exponent is alpha0=2 for every X and the Hall second-order
% exponent is beta=1. This is a size experiment.
%
% The training probit uses X and X^2. The inference sample is split at the
% median rank of the predicted persistence probability.
    %% User settings
    thisDir = fileparts(mfilename('fullpath'));
    addpath(thisDir);

    numWorkers = 8;
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local', numWorkers);
    elseif pool.NumWorkers ~= numWorkers
        delete(pool);
        pool = parpool('local', numWorkers);
    end

    helperFile = fullfile(thisDir, 'adaptive_balanced_dcov_probit_second_order.m');
    if exist(helperFile, 'file') ~= 2
        error('adaptive_balanced_dcov_probit_second_order.m must be in the same folder.');
    end
    addAttachedFiles(pool, helperFile);

    baseSeed = 2026100811;

    nList = [2000, 5000];
    kByN  = [100,  50; ...
             200, 100];

    MC     = 1000;
    Rperm  = 299;
    level  = 0.05;

    cfg = struct( ...
        'trainFraction', 1/5, ...
        'M1Fraction', 0.30, ...
        'M2Fraction', 0.25, ...
        'Rperm', Rperm);

    fprintf('DGP 1: adaptive dCov with quadratic probit score\n');
    fprintf('alpha0=2, beta=1; predictors=(X,X^2)\n');
    fprintf('Training: 20%% split; M1=30%% of training; M2=7.5%% of training\n');
    fprintf('n = 2000, 5000 only; MC=%d, permutations=%d\n\n', MC, Rperm);

    nDesigns = numel(nList)*size(kByN,2);

    N_col            = zeros(nDesigns,1);
    K_col            = zeros(nDesigns,1);
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
    Perm_col         = Rperm*ones(nDesigns,1);
    Level_col        = level*ones(nDesigns,1);

    replications = cell(nDesigns,1);
    row = 0;
    tic;

    for in = 1:numel(nList)
        n = nList(in);

        for ik = 1:size(kByN,2)
            k = kByN(in,ik);
            row = row + 1;
            designID = row;

            rejected   = false(MC,1);
            pvalues    = NaN(MC,1);
            failed     = false(MC,1);
            w1Rep      = NaN(MC,1);
            w2Rep      = NaN(MC,1);
            devRep     = NaN(MC,1);
            betaRep    = NaN(MC,3);
            scoreSdRep = NaN(MC,1);
            localWRep  = NaN(MC,2);

            fprintf('Running n=%d, k=%d ...\n', n, k);

            parfor rep = 1:MC
                rng(baseSeed + 100000*designID + rep, 'twister');

                X = 2*rand(n,1) - 1;
                U = rand(n,1);
                U(U == 0) = realmin;

                s = exp(X/2);
                Y = s.*(U.^(-1/2) - 1);

                try
                    [pval, info] = adaptive_balanced_dcov_probit_second_order(X, Y, k, cfg);
                    pvalues(rep) = pval;
                    rejected(rep) = (pval <= level);
                    w1Rep(rep) = info.training_w1;
                    w2Rep(rep) = info.training_w2;
                    devRep(rep) = info.probit_deviance;
                    betaRep(rep,:) = info.probit_beta.';
                    scoreSdRep(rep) = info.score_sd;
                    localWRep(rep,:) = info.local_w.';
                catch
                    failed(rep) = true;
                end
            end

            used = ~failed & isfinite(pvalues);
            nUsed = sum(used);

            mTrain = floor(cfg.trainFraction*n);
            M1 = floor(cfg.M1Fraction*mTrain);
            M2 = floor(cfg.M2Fraction*M1);

            N_col(row)      = n;
            K_col(row)      = k;
            TrainN_col(row) = mTrain;
            InferN_col(row) = n - mTrain;
            M1_col(row)     = M1;
            M2_col(row)     = M2;
            Fail_col(row)   = MC - nUsed;

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
                'n',n,'k',k,'pvalue',pvalues,'reject',rejected, ...
                'failed',failed,'training_w1',w1Rep,'training_w2',w2Rep, ...
                'probit_deviance',devRep,'probit_beta',betaRep, ...
                'score_sd',scoreSdRep,'local_w_hat',localWRep);

            fprintf('  rejection rate=%.4f, MCSE=%.4f, failures=%d/%d\n', ...
                Reject_col(row), MCSE_col(row), Fail_col(row), MC);
        end
    end

    toc;

    results = table( ...
        N_col,K_col,TrainN_col,InferN_col,M1_col,M2_col, ...
        Reject_col,MCSE_col,Fail_col,MeanW1_col,MeanW2_col, ...
        MeanDev_col,MeanBetaNorm_col,MeanScoreSd_col, ...
        MeanAvgW_col,MeanMinW_col,MeanMaxW_col,Perm_col,Level_col, ...
        'VariableNames', { ...
        'n','k','training_n','inference_n','M1','M2', ...
        'rejection_rate','mcse','failures','mean_training_w1', ...
        'mean_training_w2','mean_probit_deviance','mean_beta_norm', ...
        'mean_score_sd','mean_average_local_w_hat', ...
        'mean_min_local_w_hat','mean_max_local_w_hat', ...
        'permutations','nominal_level'});

    disp(results);
    writetable(results,'DGP1_dcov_adaptive_probit_30_075_summary.xlsx','Sheet','Summary');
    writetable(results,'DGP1_dcov_adaptive_probit_30_075_results.csv');

    settings = table( ...
        MC, Rperm, level, numWorkers, baseSeed, ...
        cfg.trainFraction, cfg.M1Fraction, cfg.M2Fraction, ...
        'VariableNames', { ...
        'MC_replications','permutations','nominal_level', ...
        'parallel_workers','base_seed','training_fraction', ...
        'M1_fraction_of_training','M2_fraction_of_M1'});

    writetable(settings,'DGP1_dcov_adaptive_probit_30_075_summary.xlsx','Sheet','Settings');

    save('DGP1_dcov_adaptive_probit_30_075_replications.mat', ...
        'replications','results','settings','nList','kByN', ...
        'MC','Rperm','level','cfg','numWorkers','baseSeed','-v7.3');

    figure;
    hold on;
    for in = 1:numel(nList)
        rows = (results.n == nList(in));
        [kPlot,ord] = sort(results.k(rows));
        rPlot = results.rejection_rate(rows);
        plot(kPlot,rPlot(ord),'-o','LineWidth',1.3, ...
            'DisplayName',sprintf('n=%d',nList(in)));
    end
    yline(level,'--','Nominal 5%','LineWidth',1.1);
    xlabel('Tail sample size k');
    ylabel('Rejection probability');
    title('DGP 1: adaptive quadratic-probit dCov size');
    legend('Location','best');
    grid on;
    hold off;
end
