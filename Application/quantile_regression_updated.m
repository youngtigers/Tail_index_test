%% ==============================================================
% Quantile regression for French MTPL data
%
% Covariates:
%   ClaimAmount ~ BonusMalus + VehPower
%
% Quantiles:
%   tau = 0.95 and 0.975
%
% These correspond to the two tail fractions used in the application:
%   k/n = 0.05 and 0.025.
%
% 95% confidence intervals by pairs bootstrap.
% ==============================================================

clear; clc;

rng(20260915,'twister');

%% Settings

dataFile = 'freMTPL2_application.csv';

tauList = [0.95, 0.975];

B = 199;          % bootstrap replications
alpha = 0.05;     % 95% confidence intervals

%% Parallel pool

numWorkers = 8;

pool = gcp('nocreate');

if isempty(pool)
    parpool('local',numWorkers);
elseif pool.NumWorkers ~= numWorkers
    delete(pool);
    parpool('local',numWorkers);
end

%% Read data

dat = readtable(dataFile);

requiredVars = {'ClaimAmount','BonusMalus','VehPower'};

for j = 1:numel(requiredVars)
    if ~ismember(requiredVars{j},dat.Properties.VariableNames)
        error('Variable %s is missing from the data.',requiredVars{j});
    end
end

Y = dat.ClaimAmount;

Xraw = [ ...
    dat.BonusMalus, ...
    dat.VehPower ...
    ];

%% Remove missing / invalid observations

valid = ...
    isfinite(Y) & ...
    Y > 0 & ...
    all(isfinite(Xraw),2);

Y = Y(valid);
Xraw = Xraw(valid,:);

n = length(Y);

fprintf('Number of usable positive claims = %d\n',n);

%% Standardize X
%
% Each slope therefore corresponds to a one-standard-deviation increase
% in the corresponding covariate.

muX = mean(Xraw,1);
sdX = std(Xraw,0,1);

if any(sdX <= 0)
    error('At least one covariate has zero variance.');
end

Xstd = (Xraw-muX)./sdX;

X = [ones(n,1),Xstd];

names = { ...
    'Intercept', ...
    'BonusMalus', ...
    'VehPower' ...
    };

p = size(X,2);
nTau = numel(tauList);

%% Storage

Estimate = NaN(p,nTau);
CILower  = NaN(p,nTau);
CIUpper  = NaN(p,nTau);

bootBeta = cell(nTau,1);

%% ==============================================================
% Main loop over quantiles
% ==============================================================

for it = 1:nTau

    tau = tauList(it);

    fprintf('\n====================================\n');
    fprintf('Quantile regression: tau = %.3f\n',tau);
    fprintf('ClaimAmount ~ BonusMalus + VehPower\n');
    fprintf('====================================\n');

    %% Point estimate

    beta = quantreg_lp(X,Y,tau);

    for j = 1:p
        fprintf('%-15s %12.4f\n',names{j},beta(j));
    end

    %% Pairs bootstrap

    bootBetaTau = NaN(B,p);

    fprintf('\nBootstrapping tau = %.3f ...\n',tau);

    parfor b = 1:B

        ind = randi(n,n,1);

        Yb = Y(ind);
        Xb = X(ind,:);

        try
            bootBetaTau(b,:) = quantreg_lp(Xb,Yb,tau)';
        catch
            bootBetaTau(b,:) = NaN;
        end
    end

    %% Percentile confidence intervals

    CI = NaN(p,2);

    for j = 1:p

        bj = bootBetaTau(:,j);
        bj = bj(isfinite(bj));

        if numel(bj) < max(20,ceil(0.5*B))
            warning('Few successful bootstrap draws for %s at tau=%.3f.', ...
                names{j},tau);
        end

        CI(j,:) = quantile( ...
            bj, ...
            [alpha/2,1-alpha/2] ...
            );
    end

    Estimate(:,it) = beta;
    CILower(:,it)  = CI(:,1);
    CIUpper(:,it)  = CI(:,2);

    bootBeta{it} = bootBetaTau;

    %% Display table for this quantile

    result_tau = table( ...
        string(names)', ...
        beta, ...
        CI(:,1), ...
        CI(:,2), ...
        'VariableNames',{ ...
        'Variable', ...
        'Estimate', ...
        'CI_Lower', ...
        'CI_Upper' ...
        });

    disp(result_tau);

end

%% ==============================================================
% Combined results table
% ==============================================================

Variable = strings(p*nTau,1);
Tau      = NaN(p*nTau,1);
BetaHat  = NaN(p*nTau,1);
Lower95  = NaN(p*nTau,1);
Upper95  = NaN(p*nTau,1);

row = 0;

for it = 1:nTau

    for j = 1:p

        row = row+1;

        Variable(row) = string(names{j});
        Tau(row)      = tauList(it);
        BetaHat(row)  = Estimate(j,it);
        Lower95(row)  = CILower(j,it);
        Upper95(row)  = CIUpper(j,it);
    end
end

results = table( ...
    Tau, ...
    Variable, ...
    BetaHat, ...
    Lower95, ...
    Upper95, ...
    'VariableNames',{ ...
    'tau', ...
    'Variable', ...
    'Estimate', ...
    'CI_Lower', ...
    'CI_Upper' ...
    });

fprintf('\n====================================\n');
fprintf('FINAL QUANTILE REGRESSION RESULTS\n');
fprintf('====================================\n\n');

disp(results);

%% Save results

writetable( ...
    results, ...
    'quantile_regression_application.xlsx', ...
    'Sheet','Combined');

for it = 1:nTau

    tau = tauList(it);

    rows = abs(results.tau-tau) < 1e-12;

    sheetName = sprintf('tau_%g',1000*tau);

    writetable( ...
        results(rows,:), ...
        'quantile_regression_application.xlsx', ...
        'Sheet',sheetName);
end

writetable( ...
    results, ...
    'quantile_regression_application.csv');

save( ...
    'quantile_regression_application_bootstrap.mat', ...
    'Estimate','CILower','CIUpper','bootBeta', ...
    'tauList','B','alpha','muX','sdX','names');

%% ==============================================================
% Local function: quantile regression by linear programming
% ==============================================================

function beta = quantreg_lp(X,y,tau)

% Solve
%
%   min_beta sum rho_tau(y_i-x_i' beta),
%
% where
%
%   rho_tau(u)=u*(tau-I(u<0)).
%
% Write
%
%   y-X beta = u-v,
%
% with u>=0 and v>=0.

[n,p] = size(X);

% Decision variables:
% theta = [beta; u; v].

f = [ ...
    zeros(p,1); ...
    tau*ones(n,1); ...
    (1-tau)*ones(n,1) ...
    ];

% Equality constraint:
% X beta + u - v = y.

Aeq = [ ...
    X, ...
    eye(n), ...
    -eye(n) ...
    ];

beq = y;

% beta unrestricted; u,v >= 0.

lb = [ ...
    -inf(p,1); ...
    zeros(2*n,1) ...
    ];

ub = [];

options = optimoptions( ...
    'linprog', ...
    'Display','none' ...
    );

theta = linprog( ...
    f, ...
    [],[], ...
    Aeq,beq, ...
    lb,ub, ...
    options ...
    );

if isempty(theta)
    error('linprog did not return a solution.');
end

beta = theta(1:p);

end
