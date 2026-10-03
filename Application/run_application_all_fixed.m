function combined = run_application_all()
% RUN_APPLICATION_ALL
% Master runner for the French MTPL application.
%
% Uses only:
%   X = (BonusMalus, VehPower)
%
% Runs:
%   1. Global dCov
%   2. Local 3 x 3 dCov
%   3. Kinsvater-Fried (2017) L-test
%
% Common tail sizes:
%   k_1 = round(0.05*n)
%   k_2 = round(0.025*n)
%
% Run Wang-Li separately using application_WangLi2013.m because it is
% substantially more computationally intensive.

clear; clc; close all;

dataFile = 'freMTPL2_application.csv';
kFracList = [0.05,0.025];
Rperm = 999;
level = 0.05;

dcov = application_dcov_global_local(dataFile,kFracList,Rperm,level);
kf = application_KF_Ltest(dataFile,kFracList,level);

combined = table( ...
    dcov.k, ...
    dcov.k_over_n, ...
    dcov.equivalent_threshold_quantile, ...
    dcov.global_dcov2, ...
    dcov.global_pvalue, ...
    dcov.local_dcov2, ...
    dcov.local_pvalue, ...
    kf.effective_tail_n, ...
    kf.KF_wald, ...
    kf.KF_pvalue, ...
    'VariableNames',{ ...
    'k','k_over_n','equivalent_threshold_quantile', ...
    'global_dcov2','global_dcov_pvalue', ...
    'local_dcov2','local_dcov_pvalue', ...
    'KF_effective_tail_n','KF_wald','KF_pvalue'});

fprintf('\n============================================\n');
fprintf('COMBINED APPLICATION RESULTS\n');
fprintf('============================================\n\n');
disp(combined);

writetable(combined,'application_all_methods_results.xlsx','Sheet','Results');
writetable(combined,'application_all_methods_results.csv');

%% P-value figure
figure;

% MATLAB requires xticks to be strictly increasing.
% Our default tail fractions are [0.05, 0.025], so sort them first
% and apply the same ordering to all plotted p-values.
[xplot,ord] = sort(combined.k_over_n,'ascend');

plot(xplot,combined.global_dcov_pvalue(ord),'-o', ...
    'LineWidth',1.5,'MarkerSize',7,'DisplayName','Global dCov');
hold on;

plot(xplot,combined.local_dcov_pvalue(ord),'-s', ...
    'LineWidth',1.5,'MarkerSize',7,'DisplayName','Local dCov (3 x 3)');

plot(xplot,combined.KF_pvalue(ord),'-^', ...
    'LineWidth',1.5,'MarkerSize',7,'DisplayName','Kinsvater-Fried L-test');

yline(level,'--','5% level','LineWidth',1.1);

xlabel('Tail fraction k/n');
ylabel('p-value');
title('French MTPL: tail-index homogeneity');
xticks(xplot);
ylim([0 1]);
legend('Location','best');
grid on;
hold off;

exportgraphics(gcf,'application_all_methods_pvalues.pdf','ContentType','vector');

end
