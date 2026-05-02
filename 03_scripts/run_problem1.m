%% run_problem1.m
% ============================================================
% 问题一最终版：入口条件、操作参数与出口浓度关系分析
% ============================================================
% 本脚本整合两条路线：
%   A. 可解释路线：
%      1) 直接预测 C_out：多元线性回归 + 二阶多项式特征 + Ridge 正则化
%      2) 峰值识别：逻辑回归 + Ridge 正则化
%      3) 特征重要性：Pearson/Spearman 相关系数 + 偏相关系数
%
%   B. 机理非线性路线：
%      1) 构造综合除尘强度 K = ln(C_in_mgNm3 / C_out_mgNm3)
%      2) 使用 CatBoostRegressor 拟合 K
%      3) 使用 CatBoostClassifier 识别 PeakCandidate
%
% 重要约定：
%   1) 逻辑回归和 CatBoost 分类模型使用 PeakCandidate，样本量更充足；
%   2) 振打周期分组峰值率使用 PeakEvent，即经过最小间隔抑制后的峰值事件；
%   3) C_out 连续值的 R2 接近 0 时，不强行美化，作为"平均浓度稳定、线性解释弱"的结论；
%   4) K + CatBoost 用于说明除尘强度的非线性机理关系，不把 K 反推 C_out 的 R2 作为主结论。
%
% 文件位置：
%   03_scripts/run_problem1.m
%
% 运行方式：
%   在 MATLAB 命令行运行：
%   run("03_scripts/run_problem1.m")
% ============================================================

clear; clc; close all;

%% ============================================================
% 0. 工程路径初始化
%% ============================================================

scriptPath = mfilename('fullpath');
scriptDir = fileparts(scriptPath);
projectRoot = fileparts(scriptDir);

cd(projectRoot);
addpath(genpath(projectRoot));

rawDir       = fullfile(projectRoot, "00_data", "raw");
processedDir = fullfile(projectRoot, "00_data", "processed");
resultDir    = fullfile(projectRoot, "04_results", "problem1");
figDir       = fullfile(resultDir, "figures");
tableDir     = fullfile(resultDir, "tables");
modelDir     = fullfile(resultDir, "models");
logDir       = fullfile(resultDir, "logs");

make_dir_if_not_exist(rawDir);
make_dir_if_not_exist(processedDir);
make_dir_if_not_exist(resultDir);
make_dir_if_not_exist(figDir);
make_dir_if_not_exist(tableDir);
make_dir_if_not_exist(modelDir);
make_dir_if_not_exist(logDir);

rng(2026);

print_title("问题一最终版：线性正则化 + K-CatBoost + 峰值风险综合分析");

fprintf("工程根目录：%s\n", projectRoot);
fprintf("原始数据目录：%s\n", rawDir);
fprintf("脚本文件：%s\n", fullfile(scriptDir, "run_problem1.m"));
fprintf("结果目录：%s\n", resultDir);

%% ============================================================
% 1. 读取原始数据
%% ============================================================

print_title("1. 读取原始数据");

dataFile = find_data_file(rawDir);
fprintf("读取文件：%s\n", dataFile);

T = readtable(dataFile, "VariableNamingRule", "preserve");
T = standardize_variable_names(T);

requiredVars = {'timestamp','Temp_C','C_in_gNm3','Q_Nm3h', ...
    'U1_kV','U2_kV','U3_kV','U4_kV', ...
    'T1_s','T2_s','T3_s','T4_s','C_out_mgNm3','P_total_kW'};

check_required_vars(T, requiredVars);

fprintf("原始样本数：%d\n", height(T));
disp("字段名：");
disp(T.Properties.VariableNames');

%% ============================================================
% 2. 数据清洗
%% ============================================================

print_title("2. 数据清洗");

if ~isdatetime(T.timestamp)
    try
        T.timestamp = datetime(T.timestamp);
    catch
        warning("timestamp 无法自动转为 datetime，将保持原格式。");
    end
end

numericVars = setdiff(requiredVars, {'timestamp'});
for i = 1:numel(numericVars)
    v = numericVars{i};
    if ~isnumeric(T.(v))
        T.(v) = str2double(string(T.(v)));
    else
        T.(v) = double(T.(v));
    end
end

if isdatetime(T.timestamp)
    [~, ia] = unique(T.timestamp, "stable");
    T = T(ia,:);
    T = sortrows(T, "timestamp");
end

% 输入变量必须有效，否则无法建模
inputVars = {'Temp_C','C_in_gNm3','Q_Nm3h', ...
    'U1_kV','U2_kV','U3_kV','U4_kV', ...
    'T1_s','T2_s','T3_s','T4_s'};

T = rmmissing(T, "DataVariables", inputVars);

valid = true(height(T),1);
valid = valid & T.Temp_C > 0 & T.Temp_C < 300;
valid = valid & T.C_in_gNm3 > 0 & T.C_in_gNm3 < 500;
valid = valid & T.Q_Nm3h > 0;
valid = valid & T.U1_kV > 0 & T.U1_kV < 200;
valid = valid & T.U2_kV > 0 & T.U2_kV < 200;
valid = valid & T.U3_kV > 0 & T.U3_kV < 200;
valid = valid & T.U4_kV > 0 & T.U4_kV < 200;
valid = valid & T.T1_s > 0 & T.T1_s < 2000;
valid = valid & T.T2_s > 0 & T.T2_s < 2000;
valid = valid & T.T3_s > 0 & T.T3_s < 2000;
valid = valid & T.T4_s > 0 & T.T4_s < 2000;

T = T(valid,:);

% 出口浓度和电耗不强制删行，只将异常值设为缺失
T.C_out_mgNm3(T.C_out_mgNm3 <= 0 | T.C_out_mgNm3 > 1000) = NaN;
T.P_total_kW(T.P_total_kW <= 0) = NaN;

fprintf("清洗后样本数：%d\n", height(T));
fprintf("有效出口浓度样本数：%d\n", sum(~isnan(T.C_out_mgNm3)));
fprintf("出口浓度缺失样本数：%d\n", sum(isnan(T.C_out_mgNm3)));

save(fullfile(processedDir, "P1_cleaned_data.mat"), "T");

%% ============================================================
% 3. 构造机理特征与动态特征
%% ============================================================

print_title("3. 构造机理特征与动态特征");

% 单位统一：入口 g/Nm3 -> mg/Nm3
T.C_in_mgNm3 = 1000 * T.C_in_gNm3;

epsC = 1e-9;
T.ValidK = ~isnan(T.C_in_mgNm3) & ~isnan(T.C_out_mgNm3) & ...
           T.C_in_mgNm3 > 0 & T.C_out_mgNm3 > 0;

T.K = NaN(height(T),1);
T.K(T.ValidK) = log((T.C_in_mgNm3(T.ValidK) + epsC) ./ ...
                    (T.C_out_mgNm3(T.ValidK) + epsC));

% 机理组合特征
T.DustLoad = T.C_in_gNm3 .* T.Q_Nm3h;
T.U_sum = T.U1_kV + T.U2_kV + T.U3_kV + T.U4_kV;
T.U_mean = T.U_sum / 4;
T.U_sqsum = T.U1_kV.^2 + T.U2_kV.^2 + T.U3_kV.^2 + T.U4_kV.^2;
T.U_front = (T.U1_kV + T.U2_kV) / 2;
T.U_back  = (T.U3_kV + T.U4_kV) / 2;
T.U_ratio_front_back = T.U_front ./ max(T.U_back, eps);

T.T_mean = (T.T1_s + T.T2_s + T.T3_s + T.T4_s) / 4;
T.T_front = (T.T1_s + T.T2_s) / 2;
T.T_back  = (T.T3_s + T.T4_s) / 2;
T.T_diff_back_front = T.T_back - T.T_front;

T.LoadPressure = T.DustLoad ./ max(T.U_sum, eps);

% 差分特征
for i = 1:4
    T.(sprintf("dU%d", i)) = [NaN; diff(T.(sprintf("U%d_kV", i)))];
    T.(sprintf("dT%d", i)) = [NaN; diff(T.(sprintf("T%d_s", i)))];
end

% 滞后特征：只使用过去值，避免当前值泄漏
lags = [1,3,5,10,15,30];
for L = lags
    T.("Cout_lag" + L) = lag_vector(T.C_out_mgNm3, L);
    T.("K_lag" + L) = lag_vector(T.K, L);
end

% 滚动统计：先滞后一位，再滚动
window = 30;
CoutLag1 = lag_vector(T.C_out_mgNm3, 1);
KLag1 = lag_vector(T.K, 1);

T.Cout_roll_mean30 = movmean(CoutLag1, [window-1 0], "omitnan");
T.Cout_roll_std30  = movstd(CoutLag1, [window-1 0], "omitnan");
T.K_roll_mean30 = movmean(KLag1, [window-1 0], "omitnan");
T.K_roll_std30  = movstd(KLag1, [window-1 0], "omitnan");

fprintf("有效 K 样本数：%d\n", sum(T.ValidK));

%% ============================================================
% 4. 构造瞬时排放峰值标签
%% ============================================================

print_title("4. 构造瞬时排放峰值标签：峰值点 + 峰值事件");

% 峰值点 PeakCandidate：
%   当前 C_out 相对过去60分钟局部中位数明显偏高，采用残差前3%。
% 峰值事件 PeakEvent：
%   对 PeakCandidate 做5分钟最小间隔抑制，避免连续峰重复计数。

windowPeak = 60;
targetPeakRatio = 0.03;
minGap = 5;

Cout = T.C_out_mgNm3;
CoutLag1 = lag_vector(Cout, 1);

T.Cout_base60 = movmedian(CoutLag1, [windowPeak-1 0], "omitnan");
T.Cout_mean60 = movmean(CoutLag1, [windowPeak-1 0], "omitnan");
T.Cout_std60  = movstd(CoutLag1, [windowPeak-1 0], "omitnan");
T.Cout_residual60 = T.C_out_mgNm3 - T.Cout_base60;
T.dCout = [NaN; diff(T.C_out_mgNm3)];

absDev = abs(CoutLag1 - T.Cout_base60);
T.Cout_mad60 = 1.4826 * movmedian(absDev, [windowPeak-1 0], "omitnan");
T.Cout_robust_z60 = T.Cout_residual60 ./ max(T.Cout_mad60, 1e-6);

T.PeakValid = ~isnan(T.C_out_mgNm3) & ~isnan(T.Cout_base60) & ...
              ~isnan(T.Cout_residual60) & ~isnan(T.Cout_robust_z60);

residValid = T.Cout_residual60(T.PeakValid);
zValid = T.Cout_robust_z60(T.PeakValid);

residThreshold = quantile(residValid, 1 - targetPeakRatio);
zThreshold = max(1.5, quantile(zValid, 1 - targetPeakRatio));

T.PeakCandidate = false(height(T),1);
T.PeakCandidate(T.PeakValid) = ...
    T.Cout_residual60(T.PeakValid) >= residThreshold;

% 如果希望更严格，可以启用下面联合条件。
% 但最终版本为了保持约3%峰值点，用残差分位数作为主定义。
% T.PeakCandidate(T.PeakValid) = ...
%     T.Cout_residual60(T.PeakValid) >= residThreshold & ...
%     T.Cout_robust_z60(T.PeakValid) >= zThreshold;

T.PeakEvent = suppress_close_peaks(T.PeakCandidate, T.Cout_residual60, minGap);
T.Peak = T.PeakEvent;  % 兼容后续函数，Peak 表示事件级峰值

fprintf("峰值可判定样本数：%d\n", sum(T.PeakValid));
fprintf("峰值点 PeakCandidate 数：%d\n", sum(T.PeakCandidate));
fprintf("峰值点占比：%.4f%%\n", 100 * mean(T.PeakCandidate(T.PeakValid)));
fprintf("峰值事件 PeakEvent 数：%d\n", sum(T.PeakEvent));
fprintf("峰值事件占比：%.4f%%\n", 100 * mean(T.PeakEvent(T.PeakValid)));
fprintf("峰值残差阈值：%.4f mg/Nm3\n", residThreshold);
fprintf("峰值鲁棒z阈值：%.4f\n", zThreshold);

peakInfo = table(sum(T.PeakValid), sum(T.PeakCandidate), ...
    100*mean(T.PeakCandidate(T.PeakValid)), ...
    sum(T.PeakEvent), 100*mean(T.PeakEvent(T.PeakValid)), ...
    residThreshold, zThreshold, minGap, ...
    'VariableNames', {'ValidCount','PeakPointCount','PeakPointRatio_percent', ...
    'PeakEventCount','PeakEventRatio_percent','ResidualThreshold','RobustZThreshold','MinGap_min'});

writetable(peakInfo, fullfile(tableDir, "P1_Peak_Definition_Info.csv"));

%% ============================================================
% 5. 定义特征集合
%% ============================================================

print_title("5. 定义建模特征");

staticFeatureNames = {
    'Temp_C','C_in_gNm3','Q_Nm3h', ...
    'U1_kV','U2_kV','U3_kV','U4_kV', ...
    'T1_s','T2_s','T3_s','T4_s', ...
    'DustLoad','U_sum','U_mean','U_sqsum', ...
    'U_front','U_back','U_ratio_front_back', ...
    'T_mean','T_front','T_back','T_diff_back_front', ...
    'LoadPressure'
};

dynamicFeatureNames = {
    'Temp_C','C_in_gNm3','Q_Nm3h', ...
    'U1_kV','U2_kV','U3_kV','U4_kV', ...
    'T1_s','T2_s','T3_s','T4_s', ...
    'DustLoad','U_sum','U_mean','U_sqsum', ...
    'U_front','U_back','U_ratio_front_back', ...
    'T_mean','T_front','T_back','T_diff_back_front', ...
    'LoadPressure', ...
    'Cout_lag1','Cout_lag3','Cout_lag5','Cout_lag10','Cout_lag15','Cout_lag30', ...
    'K_lag1','K_lag3','K_lag5','K_lag10','K_lag15','K_lag30', ...
    'dU1','dU2','dU3','dU4', ...
    'dT1','dT2','dT3','dT4', ...
    'Cout_roll_mean30','Cout_roll_std30', ...
    'K_roll_mean30','K_roll_std30'
};

fprintf("静态解释特征数：%d\n", numel(staticFeatureNames));
fprintf("动态预测特征数：%d\n", numel(dynamicFeatureNames));

%% ============================================================
% 6. 直接 C_out 回归：Ridge 多项式模型
%% ============================================================

print_title("6. 直接 C_out 回归：Ridge 多项式模型");

regValid = ~isnan(T.C_out_mgNm3);
regData = T(regValid,:);

X_static_raw = table2array(regData(:, staticFeatureNames));
X_dynamic_raw = table2array(regData(:, dynamicFeatureNames));
yCout = regData.C_out_mgNm3;

nReg = numel(yCout);
[idxTrain, idxVal, idxTest] = time_split_indices(nReg, 0.70, 0.15, 0.15);

fprintf("回归训练集样本数：%d\n", numel(idxTrain));
fprintf("回归验证集样本数：%d\n", numel(idxVal));
fprintf("回归测试集样本数：%d\n", numel(idxTest));

yTrain = yCout(idxTrain);
yVal   = yCout(idxVal);
yTest  = yCout(idxTest);

[XsTrainImp, fillStatic] = impute_train_median(X_static_raw(idxTrain,:));
XsValImp  = impute_apply(X_static_raw(idxVal,:), fillStatic);
XsTestImp = impute_apply(X_static_raw(idxTest,:), fillStatic);

[XdTrainImp, fillDyn] = impute_train_median(X_dynamic_raw(idxTrain,:));
XdValImp  = impute_apply(X_dynamic_raw(idxVal,:), fillDyn);
XdTestImp = impute_apply(X_dynamic_raw(idxTest,:), fillDyn);

degree = 2;
lambdaGrid = logspace(-6, 4, 60);

% 静态模型
[XsTrainPoly, polyStatic] = build_poly_train(XsTrainImp, staticFeatureNames, degree);
XsValPoly  = build_poly_apply(XsValImp, polyStatic);
XsTestPoly = build_poly_apply(XsTestImp, polyStatic);

[XsTrainStd, stdStatic] = standardize_train(XsTrainPoly);
XsValStd  = standardize_apply(XsValPoly, stdStatic);
XsTestStd = standardize_apply(XsTestPoly, stdStatic);

[betaStatic, bestLambdaStatic] = train_ridge_with_validation(XsTrainStd, yTrain, XsValStd, yVal, lambdaGrid);
yTestPredStatic = predict_ridge(XsTestStd, betaStatic);
metricsStatic = regression_metrics(yTest, yTestPredStatic);

% 动态模型
[XdTrainPoly, polyDyn] = build_poly_train(XdTrainImp, dynamicFeatureNames, degree);
XdValPoly  = build_poly_apply(XdValImp, polyDyn);
XdTestPoly = build_poly_apply(XdTestImp, polyDyn);

[XdTrainStd, stdDyn] = standardize_train(XdTrainPoly);
XdValStd  = standardize_apply(XdValPoly, stdDyn);
XdTestStd = standardize_apply(XdTestPoly, stdDyn);

[betaDyn, bestLambdaDyn] = train_ridge_with_validation(XdTrainStd, yTrain, XdValStd, yVal, lambdaGrid);
yTestPredDyn = predict_ridge(XdTestStd, betaDyn);
metricsDyn = regression_metrics(yTest, yTestPredDyn);

% 基准模型
yMeanPred = mean(yTrain, "omitnan") * ones(size(yTest));
metricsMean = regression_metrics(yTest, yMeanPred);

yPersist = regData.Cout_lag1(idxTest);
yPersist(isnan(yPersist)) = mean(yTrain, "omitnan");
metricsPersist = regression_metrics(yTest, yPersist);

fprintf("\n================ C_out 回归模型对比 ================\n");
modelCompareTable = [
    add_model_name(struct2table(metricsMean), "MeanBaseline");
    add_model_name(struct2table(metricsPersist), "PersistenceBaseline");
    add_model_name(struct2table(metricsStatic), "StaticRidgePoly");
    add_model_name(struct2table(metricsDyn), "DynamicRidgePoly")
];
disp(modelCompareTable);
fprintf("静态模型最佳 lambda：%.6g\n", bestLambdaStatic);
fprintf("动态模型最佳 lambda：%.6g\n", bestLambdaDyn);

writetable(modelCompareTable, fullfile(tableDir, "P1_Cout_Regression_Model_Comparison.csv"));

predTable = table(regData.timestamp(idxTest), yTest, yTestPredStatic, yTestPredDyn, yPersist, ...
    'VariableNames', {'timestamp','Cout_True','Cout_Pred_Static','Cout_Pred_Dynamic','Cout_Pred_Persistence'});
writetable(predTable, fullfile(tableDir, "P1_Cout_Test_Predictions.csv"));

plot_prediction(yTest, yTestPredStatic, ...
    "静态Ridge多项式模型：出口浓度真实值与预测值", ...
    "测试集样本序号", "C_{out} / mg·Nm^{-3}", ...
    fullfile(figDir, "P1_Cout_Static_Prediction.png"));

plot_prediction(yTest, yTestPredDyn, ...
    "动态Ridge多项式模型：出口浓度真实值与预测值", ...
    "测试集样本序号", "C_{out} / mg·Nm^{-3}", ...
    fullfile(figDir, "P1_Cout_Dynamic_Prediction.png"));

plot_scatter(yTest, yTestPredDyn, ...
    "动态Ridge多项式模型：出口浓度真实值-预测值散点图", ...
    "真实 C_{out}", "预测 C_{out}", ...
    fullfile(figDir, "P1_Cout_Dynamic_Scatter.png"));

%% ============================================================
% 7. 相关系数与偏相关系数
%% ============================================================

print_title("7. 相关系数与偏相关系数分析");

controlVars = {'C_in_gNm3','Q_Nm3h','Temp_C'};

corrTable = compute_corr_importance(regData, staticFeatureNames, "C_out_mgNm3");
partialTable = compute_partial_corr_importance(regData, staticFeatureNames, "C_out_mgNm3", controlVars);

fprintf("\nPearson/Spearman 相关系数 Top 20：\n");
disp(corrTable(1:min(20,height(corrTable)),:));

fprintf("\n偏相关系数 Top 20：\n");
disp(partialTable(1:min(20,height(partialTable)),:));

writetable(corrTable, fullfile(tableDir, "P1_Corr_Importance.csv"));
writetable(partialTable, fullfile(tableDir, "P1_PartialCorr_Importance.csv"));

plot_bar_importance(corrTable.Feature, abs(corrTable.PearsonR), ...
    "C_{out} Pearson相关系数绝对值 Top 20", ...
    fullfile(figDir, "P1_PearsonCorr_Top20.png"), 20);

plot_bar_importance(partialTable.Feature, abs(partialTable.PartialR), ...
    "C_{out} 偏相关系数绝对值 Top 20", ...
    fullfile(figDir, "P1_PartialCorr_Top20.png"), 20);

%% ============================================================
% 8. Ridge 逻辑回归识别峰值点 PeakCandidate
%% ============================================================

print_title("8. Ridge 逻辑回归识别峰值点 PeakCandidate");

clsValid = T.PeakValid;
clsData = T(clsValid,:);
X_cls_raw = table2array(clsData(:, dynamicFeatureNames));

% 最终版关键点：分类模型使用 PeakCandidate，而不是 PeakEvent
yPeakPoint = double(clsData.PeakCandidate);

[idxTrainC, idxValC, idxTestC] = stratified_split_indices(yPeakPoint, 0.70, 0.15, 0.15, 2026);

fprintf("分类训练集样本数：%d，峰值点比例：%.4f%%，峰值点数：%d\n", ...
    numel(idxTrainC), 100*mean(yPeakPoint(idxTrainC)), sum(yPeakPoint(idxTrainC)));
fprintf("分类验证集样本数：%d，峰值点比例：%.4f%%，峰值点数：%d\n", ...
    numel(idxValC), 100*mean(yPeakPoint(idxValC)), sum(yPeakPoint(idxValC)));
fprintf("分类测试集样本数：%d，峰值点比例：%.4f%%，峰值点数：%d\n", ...
    numel(idxTestC), 100*mean(yPeakPoint(idxTestC)), sum(yPeakPoint(idxTestC)));

[XcTrainImp, fillCls] = impute_train_median(X_cls_raw(idxTrainC,:));
XcValImp  = impute_apply(X_cls_raw(idxValC,:), fillCls);
XcTestImp = impute_apply(X_cls_raw(idxTestC,:), fillCls);

[XcTrainStd, stdCls] = standardize_train(XcTrainImp);
XcValStd  = standardize_apply(XcValImp, stdCls);
XcTestStd = standardize_apply(XcTestImp, stdCls);

lambdaGridCls = logspace(-5, 3, 40);

[betaLogit, bestLambdaLogit, scoreValLogit] = train_logistic_ridge_with_validation( ...
    XcTrainStd, yPeakPoint(idxTrainC), XcValStd, yPeakPoint(idxValC), lambdaGridCls);

bestThresholdLogit = choose_best_threshold_f2(yPeakPoint(idxValC), scoreValLogit);

scoreTestLogit = predict_logistic(XcTestStd, betaLogit);
yPredLogit = double(scoreTestLogit >= bestThresholdLogit);

metricsLogit = classification_metrics(yPeakPoint(idxTestC), yPredLogit, scoreTestLogit);

fprintf("逻辑回归最佳 lambda：%.6g\n", bestLambdaLogit);
fprintf("逻辑回归最优阈值：%.4f\n", bestThresholdLogit);
fprintf("\n================ Ridge逻辑回归峰值点识别结果 ================\n");
disp(struct2table(metricsLogit));

writetable(struct2table(metricsLogit), fullfile(tableDir, "P1_Peak_Logistic_Metrics.csv"));

peakPredTable = table(clsData.timestamp(idxTestC), yPeakPoint(idxTestC), scoreTestLogit, yPredLogit, ...
    'VariableNames', {'timestamp','PeakPoint_True','PeakPoint_Probability','PeakPoint_Pred'});
writetable(peakPredTable, fullfile(tableDir, "P1_Peak_Logistic_Test_Predictions.csv"));

plot_peak_detection(clsData.timestamp(idxTestC), clsData.C_out_mgNm3(idxTestC), ...
    yPeakPoint(idxTestC), yPredLogit, scoreTestLogit, ...
    fullfile(figDir, "P1_Peak_Logistic_Detection.png"));

%% ============================================================
% 9. 峰值点与峰值事件相关性分析
%% ============================================================

print_title("9. 峰值点与峰值事件相关性分析");

peakPointCorrTable = compute_corr_importance(T(T.PeakValid,:), staticFeatureNames, "PeakCandidate");
peakPointPartialTable = compute_partial_corr_importance(T(T.PeakValid,:), staticFeatureNames, "PeakCandidate", controlVars);

peakEventCorrTable = compute_corr_importance(T(T.PeakValid,:), staticFeatureNames, "PeakEvent");
peakEventPartialTable = compute_partial_corr_importance(T(T.PeakValid,:), staticFeatureNames, "PeakEvent", controlVars);

fprintf("\nPeakCandidate 与静态特征相关系数 Top 20：\n");
disp(peakPointCorrTable(1:min(20,height(peakPointCorrTable)),:));

fprintf("\nPeakCandidate 与静态特征偏相关系数 Top 20：\n");
disp(peakPointPartialTable(1:min(20,height(peakPointPartialTable)),:));

fprintf("\nPeakEvent 与静态特征相关系数 Top 20：\n");
disp(peakEventCorrTable(1:min(20,height(peakEventCorrTable)),:));

fprintf("\nPeakEvent 与静态特征偏相关系数 Top 20：\n");
disp(peakEventPartialTable(1:min(20,height(peakEventPartialTable)),:));

writetable(peakPointCorrTable, fullfile(tableDir, "P1_PeakPoint_Corr_Importance.csv"));
writetable(peakPointPartialTable, fullfile(tableDir, "P1_PeakPoint_PartialCorr_Importance.csv"));
writetable(peakEventCorrTable, fullfile(tableDir, "P1_PeakEvent_Corr_Importance.csv"));
writetable(peakEventPartialTable, fullfile(tableDir, "P1_PeakEvent_PartialCorr_Importance.csv"));

%% ============================================================
% 10. K + CatBoost 非线性机理模型
%% ============================================================

print_title("10. K + CatBoost 非线性机理模型");

hasCatBoost = false;
np = [];
catboost = [];

try
    pe = pyenv;
    fprintf("当前 MATLAB Python 环境：\n");
    disp(pe);

    np = py.importlib.import_module("numpy");
    catboost = py.importlib.import_module("catboost");
    hasCatBoost = true;
    fprintf("[成功] 已导入 Python numpy 与 catboost。\n");
catch ME
    warning("[跳过 CatBoost] 无法导入 Python catboost 或 numpy：%s", ME.message);
end

metricsKCat = [];
metricsCoutFromKCat = [];
impKCat = table();
metricsPeakCat = [];
impPeakCat = table();

if hasCatBoost
    % 10.1 K 回归数据集
    kValid = T.ValidK;
    kData = T(kValid,:);

    X_k_raw = table2array(kData(:, dynamicFeatureNames));
    yK = kData.K;
    yCoutK = kData.C_out_mgNm3;
    CinK_mg = kData.C_in_mgNm3;

    nK = numel(yK);
    [idxKTrain, idxKVal, idxKTest] = time_split_indices(nK, 0.70, 0.15, 0.15);

    [XkTrain, fillK] = impute_train_median(X_k_raw(idxKTrain,:));
    XkVal  = impute_apply(X_k_raw(idxKVal,:), fillK);
    XkTest = impute_apply(X_k_raw(idxKTest,:), fillK);

    yKTrain = yK(idxKTrain);
    yKVal = yK(idxKVal);
    yKTest = yK(idxKTest);

    yCoutKTest = yCoutK(idxKTest);
    CinKTest_mg = CinK_mg(idxKTest);

    fprintf("K回归训练集：%d，验证集：%d，测试集：%d\n", numel(idxKTrain), numel(idxKVal), numel(idxKTest));

    modelK = catboost.CatBoostRegressor(pyargs( ...
        'iterations', int32(1500), ...
        'depth', int32(6), ...
        'learning_rate', 0.03, ...
        'loss_function', 'RMSE', ...
        'eval_metric', 'RMSE', ...
        'random_seed', int32(2026), ...
        'od_type', 'Iter', ...
        'od_wait', int32(100), ...
        'verbose', int32(200) ...
    ));

    modelK.fit(np.array(XkTrain), np.array(yKTrain(:)), pyargs( ...
        'eval_set', py.tuple({np.array(XkVal), np.array(yKVal(:))}), ...
        'use_best_model', true ...
    ));

    fprintf("CatBoostRegressor K模型训练完成。\n");

    KPredRaw = pyvector_to_double(modelK.predict(np.array(XkTest)));
    KPred = KPredRaw(:);

    % K 反推 C_out，注意单位已经统一为 mg/Nm3
    CoutFromKPred = CinKTest_mg(:) .* exp(-KPred(:));

    metricsKCat = regression_metrics(yKTest, KPred);
    metricsCoutFromKCat = regression_metrics(yCoutKTest, CoutFromKPred);

    fprintf("\n================ CatBoost K预测评估结果 ================\n");
    disp(struct2table(metricsKCat));

    fprintf("\n================ CatBoost K反推C_out辅助评估结果 ================\n");
    disp(struct2table(metricsCoutFromKCat));

    writetable(struct2table(metricsKCat), fullfile(tableDir, "P1_CatBoost_K_Metrics.csv"));
    writetable(struct2table(metricsCoutFromKCat), fullfile(tableDir, "P1_CatBoost_Cout_FromK_Metrics.csv"));

    try
        modelK.save_model(char(fullfile(modelDir, "P1_CatBoost_K_Model.cbm")));
    catch
        warning("CatBoost K模型保存失败。");
    end

    % 特征重要性
    impK = pyvector_to_double(modelK.get_feature_importance());
    impKCat = table(dynamicFeatureNames(:), impK(:), ...
        'VariableNames', {'Feature','Importance'});
    impKCat = sortrows(impKCat, "Importance", "descend");
    writetable(impKCat, fullfile(tableDir, "P1_CatBoost_K_FeatureImportance.csv"));

    fprintf("\nCatBoost K模型特征重要性 Top 20：\n");
    disp(impKCat(1:min(20,height(impKCat)),:));

    plot_prediction(yCoutKTest, CoutFromKPred, ...
        "CatBoost K反推出口浓度：真实值与预测值", ...
        "测试集样本序号", "C_{out} / mg·Nm^{-3}", ...
        fullfile(figDir, "P1_CatBoost_Cout_FromK_Prediction.png"));

    % 10.2 CatBoost 分类识别 PeakCandidate
    XcTrainCB = X_cls_raw(idxTrainC,:);
    XcValCB = X_cls_raw(idxValC,:);
    XcTestCB = X_cls_raw(idxTestC,:);

    [XcTrainCB, fillCBCls] = impute_train_median(XcTrainCB);
    XcValCB = impute_apply(XcValCB, fillCBCls);
    XcTestCB = impute_apply(XcTestCB, fillCBCls);

    ycTrain = yPeakPoint(idxTrainC);
    ycVal = yPeakPoint(idxValC);
    ycTest = yPeakPoint(idxTestC);

    modelPeak = catboost.CatBoostClassifier(pyargs( ...
        'iterations', int32(1500), ...
        'depth', int32(5), ...
        'learning_rate', 0.03, ...
        'loss_function', 'Logloss', ...
        'eval_metric', 'AUC', ...
        'auto_class_weights', 'Balanced', ...
        'random_seed', int32(2026), ...
        'verbose', int32(200) ...
    ));

    modelPeak.fit(np.array(XcTrainCB), np.array(ycTrain(:)), pyargs( ...
        'eval_set', py.tuple({np.array(XcValCB), np.array(ycVal(:))}), ...
        'use_best_model', true, ...
        'early_stopping_rounds', int32(100) ...
    ));

    fprintf("CatBoostClassifier 峰值点模型训练完成。\n");

    scoreTestCat = catboost_predict_proba_positive(modelPeak, np.array(XcTestCB));
    scoreValCat = catboost_predict_proba_positive(modelPeak, np.array(XcValCB));

    bestThresholdCat = choose_best_threshold_f2(ycVal, scoreValCat);
    yPredCat = double(scoreTestCat >= bestThresholdCat);

    metricsPeakCat = classification_metrics(ycTest, yPredCat, scoreTestCat);

    fprintf("CatBoost峰值点最优阈值：%.4f\n", bestThresholdCat);
    fprintf("\n================ CatBoost峰值点识别结果 ================\n");
    disp(struct2table(metricsPeakCat));

    writetable(struct2table(metricsPeakCat), fullfile(tableDir, "P1_CatBoost_PeakPoint_Metrics.csv"));

    peakCatPredTable = table(clsData.timestamp(idxTestC), ycTest, scoreTestCat, yPredCat, ...
        'VariableNames', {'timestamp','PeakPoint_True','PeakPoint_Probability','PeakPoint_Pred'});
    writetable(peakCatPredTable, fullfile(tableDir, "P1_CatBoost_PeakPoint_Test_Predictions.csv"));

    try
        modelPeak.save_model(char(fullfile(modelDir, "P1_CatBoost_PeakPoint_Model.cbm")));
    catch
        warning("CatBoost峰值点模型保存失败。");
    end

    impPeak = pyvector_to_double(modelPeak.get_feature_importance());
    impPeakCat = table(dynamicFeatureNames(:), impPeak(:), ...
        'VariableNames', {'Feature','Importance'});
    impPeakCat = sortrows(impPeakCat, "Importance", "descend");
    writetable(impPeakCat, fullfile(tableDir, "P1_CatBoost_PeakPoint_FeatureImportance.csv"));

    fprintf("\nCatBoost峰值点模型特征重要性 Top 20：\n");
    disp(impPeakCat(1:min(20,height(impPeakCat)),:));

    plot_peak_detection(clsData.timestamp(idxTestC), clsData.C_out_mgNm3(idxTestC), ...
        ycTest, yPredCat, scoreTestCat, ...
        fullfile(figDir, "P1_CatBoost_PeakPoint_Detection.png"));
end

%% ============================================================
% 11. 振打周期分组峰值事件频率分析
%% ============================================================

print_title("11. 振打周期分组峰值事件频率分析");

allRapTables = table();

for v = ["T1_s","T2_s","T3_s","T4_s"]
    rapTable = rapping_peak_rate_table(T, char(v), 5, "PeakEvent");
    allRapTables = [allRapTables; rapTable]; %#ok<AGROW>
    fprintf("\n%s 振打周期分组峰值事件率：\n", v);
    disp(rapTable);
end

writetable(allRapTables, fullfile(tableDir, "P1_All_RappingPeriod_PeakRate.csv"));
plot_rapping_peak_rate(allRapTables, fullfile(figDir, "P1_RappingPeriod_PeakRate.png"));

%% ============================================================
% 12. 综合结论汇总表
%% ============================================================

print_title("12. 保存综合汇总结果");

summaryRows = {};
summaryRows(end+1,:) = {"MeanBaseline_Cout", metricsMean.MAE, metricsMean.RMSE, metricsMean.R2, metricsMean.MAPE, NaN, "C_out均值基准"};
summaryRows(end+1,:) = {"StaticRidgePoly_Cout", metricsStatic.MAE, metricsStatic.RMSE, metricsStatic.R2, metricsStatic.MAPE, NaN, "平均浓度线性解释模型"};
summaryRows(end+1,:) = {"DynamicRidgePoly_Cout", metricsDyn.MAE, metricsDyn.RMSE, metricsDyn.R2, metricsDyn.MAPE, NaN, "加入滞后项的动态预测模型"};

if ~isempty(metricsKCat)
    summaryRows(end+1,:) = {"CatBoost_K", metricsKCat.MAE, metricsKCat.RMSE, metricsKCat.R2, metricsKCat.MAPE, NaN, "综合除尘强度非线性机理模型"};
    summaryRows(end+1,:) = {"CatBoost_CoutFromK", metricsCoutFromKCat.MAE, metricsCoutFromKCat.RMSE, metricsCoutFromKCat.R2, metricsCoutFromKCat.MAPE, NaN, "K反推出口浓度辅助评估"};
end

summaryTable = cell2table(summaryRows, ...
    'VariableNames', {'Model','MAE','RMSE','R2','MAPE','AUC','Meaning'});

writetable(summaryTable, fullfile(tableDir, "P1_Model_Summary.csv"));

clsSummaryRows = {};
clsSummaryRows(end+1,:) = {"RidgeLogistic_PeakCandidate", metricsLogit.Accuracy, metricsLogit.Precision, metricsLogit.Recall, metricsLogit.F1, metricsLogit.F2, metricsLogit.AUC, metricsLogit.TP, metricsLogit.TN, metricsLogit.FP, metricsLogit.FN};

if ~isempty(metricsPeakCat)
    clsSummaryRows(end+1,:) = {"CatBoost_PeakCandidate", metricsPeakCat.Accuracy, metricsPeakCat.Precision, metricsPeakCat.Recall, metricsPeakCat.F1, metricsPeakCat.F2, metricsPeakCat.AUC, metricsPeakCat.TP, metricsPeakCat.TN, metricsPeakCat.FP, metricsPeakCat.FN};
end

clsSummaryTable = cell2table(clsSummaryRows, ...
    'VariableNames', {'Model','Accuracy','Precision','Recall','F1','F2','AUC','TP','TN','FP','FN'});

writetable(clsSummaryTable, fullfile(tableDir, "P1_Classification_Summary.csv"));

disp("回归/机理模型汇总：");
disp(summaryTable);

disp("峰值分类模型汇总：");
disp(clsSummaryTable);

% 保存模型参数
ridgeStaticModel = struct();
ridgeStaticModel.beta = betaStatic;
ridgeStaticModel.poly = polyStatic;
ridgeStaticModel.std = stdStatic;
ridgeStaticModel.fill = fillStatic;
ridgeStaticModel.featureNames = staticFeatureNames;

ridgeDynamicModel = struct();
ridgeDynamicModel.beta = betaDyn;
ridgeDynamicModel.poly = polyDyn;
ridgeDynamicModel.std = stdDyn;
ridgeDynamicModel.fill = fillDyn;
ridgeDynamicModel.featureNames = dynamicFeatureNames;

logisticPeakModel = struct();
logisticPeakModel.beta = betaLogit;
logisticPeakModel.std = stdCls;
logisticPeakModel.fill = fillCls;
logisticPeakModel.featureNames = dynamicFeatureNames;
logisticPeakModel.threshold = bestThresholdLogit;

save(fullfile(modelDir, "P1_Ridge_Static_Cout_Model.mat"), "ridgeStaticModel");
save(fullfile(modelDir, "P1_Ridge_Dynamic_Cout_Model.mat"), "ridgeDynamicModel");
save(fullfile(modelDir, "P1_LogisticRidge_PeakCandidate_Model.mat"), "logisticPeakModel");

problem1_results = struct();
problem1_results.peakInfo = peakInfo;
problem1_results.modelCompareTable = modelCompareTable;
problem1_results.summaryTable = summaryTable;
problem1_results.clsSummaryTable = clsSummaryTable;
problem1_results.corrTable = corrTable;
problem1_results.partialTable = partialTable;
problem1_results.peakPointCorrTable = peakPointCorrTable;
problem1_results.peakPointPartialTable = peakPointPartialTable;
problem1_results.peakEventCorrTable = peakEventCorrTable;
problem1_results.peakEventPartialTable = peakEventPartialTable;
problem1_results.allRapTables = allRapTables;
problem1_results.hasCatBoost = hasCatBoost;

if hasCatBoost
    problem1_results.impKCat = impKCat;
    problem1_results.impPeakCat = impPeakCat;
end

save(fullfile(resultDir, "P1_results_summary.mat"), "problem1_results");

fprintf("\n问题一最终版运行完成。\n");
fprintf("结果保存目录：%s\n", resultDir);

fprintf("\n主要输出文件：\n");
fprintf("1. %s\n", fullfile(tableDir, "P1_Cout_Regression_Model_Comparison.csv"));
fprintf("2. %s\n", fullfile(tableDir, "P1_Corr_Importance.csv"));
fprintf("3. %s\n", fullfile(tableDir, "P1_PartialCorr_Importance.csv"));
fprintf("4. %s\n", fullfile(tableDir, "P1_Peak_Definition_Info.csv"));
fprintf("5. %s\n", fullfile(tableDir, "P1_Peak_Logistic_Metrics.csv"));
fprintf("6. %s\n", fullfile(tableDir, "P1_CatBoost_K_Metrics.csv"));
fprintf("7. %s\n", fullfile(tableDir, "P1_CatBoost_PeakPoint_Metrics.csv"));
fprintf("8. %s\n", fullfile(tableDir, "P1_All_RappingPeriod_PeakRate.csv"));
fprintf("9. %s\n", fullfile(tableDir, "P1_Model_Summary.csv"));
fprintf("10. %s\n", fullfile(tableDir, "P1_Classification_Summary.csv"));
fprintf("11. %s\n", fullfile(figDir, "P1_RappingPeriod_PeakRate.png"));

%% ============================================================
% 局部函数
%% ============================================================

function print_title(str)
    fprintf("\n============================================================\n");
    fprintf("%s\n", str);
    fprintf("============================================================\n");
end

function make_dir_if_not_exist(d)
    if ~exist(d, "dir")
        mkdir(d);
    end
end

function dataFile = find_data_file(rawDir)
    preferred = fullfile(rawDir, "Cement_ESP_Data.csv");
    if exist(preferred, "file")
        dataFile = preferred;
        return;
    end

    files = [dir(fullfile(rawDir, "*.csv")); ...
             dir(fullfile(rawDir, "*.xlsx")); ...
             dir(fullfile(rawDir, "*.xls"))];

    if isempty(files)
        error("在 %s 中找不到 csv/xlsx/xls 数据文件。", rawDir);
    end

    dataFile = fullfile(files(1).folder, files(1).name);
end

function T = standardize_variable_names(T)
    names = string(T.Properties.VariableNames);
    names = strtrim(names);

    for i = 1:numel(names)
        s = names(i);
        s = replace(s, " ", "");
        s = replace(s, "（", "(");
        s = replace(s, "）", ")");
        names(i) = s;
    end

    T.Properties.VariableNames = cellstr(names);
end

function check_required_vars(T, requiredVars)
    miss = setdiff(requiredVars, T.Properties.VariableNames);
    if ~isempty(miss)
        error("数据缺少必要字段：%s", strjoin(miss, ", "));
    end
end

function y = lag_vector(x, L)
    x = x(:);
    y = NaN(size(x));
    if L < numel(x)
        y((L+1):end) = x(1:end-L);
    end
end

function [idxTrain, idxVal, idxTest] = time_split_indices(n, rTrain, rVal, rTest)
    if abs(rTrain + rVal + rTest - 1) > 1e-9
        error("划分比例之和必须为1。");
    end
    nTrain = floor(rTrain * n);
    nVal = floor(rVal * n);
    idxTrain = (1:nTrain)';
    idxVal = (nTrain+1:nTrain+nVal)';
    idxTest = (nTrain+nVal+1:n)';
end

function [idxTrain, idxVal, idxTest] = stratified_split_indices(y, rTrain, rVal, rTest, seed)
    rng(seed);
    y = y(:);

    idxPos = find(y == 1);
    idxNeg = find(y == 0);

    idxPos = idxPos(randperm(numel(idxPos)));
    idxNeg = idxNeg(randperm(numel(idxNeg)));

    nPos = numel(idxPos);
    nNeg = numel(idxNeg);

    nPosTrain = floor(rTrain * nPos);
    nPosVal = floor(rVal * nPos);
    nNegTrain = floor(rTrain * nNeg);
    nNegVal = floor(rVal * nNeg);

    idxTrain = [idxPos(1:nPosTrain); idxNeg(1:nNegTrain)];
    idxVal = [idxPos(nPosTrain+1:nPosTrain+nPosVal); idxNeg(nNegTrain+1:nNegTrain+nNegVal)];
    idxTest = [idxPos(nPosTrain+nPosVal+1:end); idxNeg(nNegTrain+nNegVal+1:end)];

    idxTrain = idxTrain(randperm(numel(idxTrain)));
    idxVal = idxVal(randperm(numel(idxVal)));
    idxTest = idxTest(randperm(numel(idxTest)));
end

function [Ximp, fillVal] = impute_train_median(X)
    fillVal = median(X, 1, "omitnan");
    fillVal(isnan(fillVal)) = 0;
    Ximp = impute_apply(X, fillVal);
end

function Ximp = impute_apply(X, fillVal)
    Ximp = X;
    for j = 1:size(Ximp,2)
        miss = isnan(Ximp(:,j));
        Ximp(miss,j) = fillVal(j);
    end
end

function [Xstd, s] = standardize_train(X)
    s.mu = mean(X, 1, "omitnan");
    s.sigma = std(X, 0, 1, "omitnan");
    s.sigma(isnan(s.sigma) | s.sigma < 1e-12) = 1;
    Xstd = (X - s.mu) ./ s.sigma;
end

function Xstd = standardize_apply(X, s)
    Xstd = (X - s.mu) ./ s.sigma;
end

function [Xpoly, polyInfo] = build_poly_train(X, featureNames, degree)
    if degree ~= 2
        error("当前仅实现二阶多项式。");
    end

    p = size(X,2);
    Xpoly = X;
    names = string(featureNames(:))';

    Xpoly = [Xpoly, X.^2];
    names = [names, names + "^2"];

    maxInteractVars = min(p, 20);
    pairs = [];
    Xinter = [];

    for i = 1:maxInteractVars
        for j = i+1:maxInteractVars
            pairs = [pairs; i, j]; %#ok<AGROW>
            Xinter = [Xinter, X(:,i).*X(:,j)]; %#ok<AGROW>
            names = [names, string(featureNames{i}) + "*" + string(featureNames{j})]; %#ok<AGROW>
        end
    end

    Xpoly = [Xpoly, Xinter];

    polyInfo.degree = degree;
    polyInfo.featureNames = featureNames;
    polyInfo.pOriginal = p;
    polyInfo.pairs = pairs;
    polyInfo.polyNames = cellstr(names);
end

function Xpoly = build_poly_apply(X, polyInfo)
    Xpoly = X;
    Xpoly = [Xpoly, X.^2];

    Xinter = [];
    pairs = polyInfo.pairs;

    for k = 1:size(pairs,1)
        i = pairs(k,1);
        j = pairs(k,2);
        Xinter = [Xinter, X(:,i).*X(:,j)]; %#ok<AGROW>
    end

    Xpoly = [Xpoly, Xinter];
end

function [beta, bestLambda] = train_ridge_with_validation(XTrain, yTrain, XVal, yVal, lambdaGrid)
    Xtr = [ones(size(XTrain,1),1), XTrain];
    Xva = [ones(size(XVal,1),1), XVal];

    p = size(Xtr,2);
    I = eye(p);
    I(1,1) = 0;

    bestRMSE = inf;
    bestLambda = NaN;
    beta = zeros(p,1);

    for lam = lambdaGrid
        b = (Xtr' * Xtr + lam * I) \ (Xtr' * yTrain);
        pred = Xva * b;
        m = regression_metrics(yVal, pred);
        if m.RMSE < bestRMSE
            bestRMSE = m.RMSE;
            beta = b;
            bestLambda = lam;
        end
    end
end

function yPred = predict_ridge(X, beta)
    yPred = [ones(size(X,1),1), X] * beta;
end

function metrics = regression_metrics(y, yPred)
    y = y(:);
    yPred = yPred(:);

    valid = ~isnan(y) & ~isnan(yPred) & isfinite(y) & isfinite(yPred);
    y = y(valid);
    yPred = yPred(valid);

    err = y - yPred;

    metrics = struct();
    metrics.MAE = mean(abs(err));
    metrics.RMSE = sqrt(mean(err.^2));
    metrics.R2 = 1 - sum(err.^2) / max(sum((y - mean(y)).^2), eps);
    metrics.MAPE = mean(abs(err) ./ max(abs(y), eps)) * 100;
end

function T2 = add_model_name(T1, modelName)
    T2 = addvars(T1, string(modelName), 'Before', 1, 'NewVariableNames', 'Model');
end

function corrTable = compute_corr_importance(T, featureNames, targetName)
    y = double(T.(targetName));
    n = numel(featureNames);

    pearsonR = NaN(n,1);
    pearsonP = NaN(n,1);
    spearmanR = NaN(n,1);
    spearmanP = NaN(n,1);

    for i = 1:n
        x = double(T.(featureNames{i}));
        valid = ~isnan(x) & ~isnan(y) & isfinite(x) & isfinite(y);
        if sum(valid) > 10
            [r,p] = corr(x(valid), y(valid), "Type", "Pearson");
            pearsonR(i) = r;
            pearsonP(i) = p;

            [r,p] = corr(x(valid), y(valid), "Type", "Spearman");
            spearmanR(i) = r;
            spearmanP(i) = p;
        end
    end

    corrTable = table(featureNames(:), pearsonR, pearsonP, spearmanR, spearmanP, ...
        abs(pearsonR), abs(spearmanR), ...
        'VariableNames', {'Feature','PearsonR','PearsonP','SpearmanR','SpearmanP','AbsPearsonR','AbsSpearmanR'});

    corrTable = sortrows(corrTable, "AbsPearsonR", "descend");
end

function partialTable = compute_partial_corr_importance(T, featureNames, targetName, controlVars)
    y = double(T.(targetName));
    n = numel(featureNames);

    C = [];
    for k = 1:numel(controlVars)
        C = [C, double(T.(controlVars{k}))]; %#ok<AGROW>
    end

    partialR = NaN(n,1);
    partialP = NaN(n,1);

    for i = 1:n
        x = double(T.(featureNames{i}));
        valid = ~isnan(x) & ~isnan(y) & all(~isnan(C),2) & ...
                isfinite(x) & isfinite(y) & all(isfinite(C),2);

        if sum(valid) > size(C,2) + 10
            xv = x(valid);
            yv = y(valid);
            Cv = C(valid,:);

            C1 = [ones(size(Cv,1),1), Cv];

            rx = xv - C1 * (C1 \ xv);
            ry = yv - C1 * (C1 \ yv);

            [r,p] = corr(rx, ry, "Type", "Pearson");
            partialR(i) = r;
            partialP(i) = p;
        end
    end

    partialTable = table(featureNames(:), partialR, partialP, abs(partialR), ...
        'VariableNames', {'Feature','PartialR','PartialP','AbsPartialR'});

    partialTable = sortrows(partialTable, "AbsPartialR", "descend");
end

function yPeak = suppress_close_peaks(candidate, score, minGap)
    candidate = candidate(:);
    score = score(:);
    yPeak = false(size(candidate));

    idx = find(candidate);
    if isempty(idx)
        return;
    end

    used = false(size(candidate));

    for k = 1:numel(idx)
        i = idx(k);
        if used(i)
            continue;
        end

        left = max(1, i - minGap);
        right = min(numel(candidate), i + minGap);
        localIdx = find(candidate(left:right)) + left - 1;

        [~, bestLocal] = max(score(localIdx));
        bestIdx = localIdx(bestLocal);

        yPeak(bestIdx) = true;
        used(localIdx) = true;
    end
end

function [beta, bestLambda, scoreValBest] = train_logistic_ridge_with_validation(XTrain, yTrain, XVal, yVal, lambdaGrid)
    Xtr = [ones(size(XTrain,1),1), XTrain];
    Xva = [ones(size(XVal,1),1), XVal];

    yTrain = yTrain(:);
    yVal = yVal(:);

    bestF2 = -inf;
    beta = zeros(size(Xtr,2),1);
    bestLambda = NaN;
    scoreValBest = zeros(size(yVal));

    for lam = lambdaGrid
        b = fit_logistic_ridge_irls(Xtr, yTrain, lam, 100);
        scoreVal = sigmoid(Xva * b);
        th = choose_best_threshold_f2(yVal, scoreVal);
        predVal = double(scoreVal >= th);
        m = classification_metrics(yVal, predVal, scoreVal);

        if m.F2 > bestF2
            bestF2 = m.F2;
            beta = b;
            bestLambda = lam;
            scoreValBest = scoreVal;
        end
    end
end

function beta = fit_logistic_ridge_irls(X, y, lambda, maxIter)
    [n,p] = size(X);
    beta = zeros(p,1);

    I = eye(p);
    I(1,1) = 0;

    posRate = mean(y);
    wClass = ones(n,1);
    if posRate > 0 && posRate < 0.5
        wClass(y == 1) = 0.5 / posRate;
        wClass(y == 0) = 0.5 / (1 - posRate);
    end

    for iter = 1:maxIter
        eta = X * beta;
        pHat = sigmoid(eta);

        W = pHat .* (1 - pHat);
        W = max(W, 1e-6);
        W = W .* wClass;

        z = eta + (y - pHat) ./ max(pHat .* (1 - pHat), 1e-6);

        XW = X .* W;
        betaNew = (XW' * X + lambda * I) \ (XW' * z);

        if norm(betaNew - beta) < 1e-6 * (1 + norm(beta))
            beta = betaNew;
            break;
        end

        beta = betaNew;
    end
end

function p = predict_logistic(X, beta)
    p = sigmoid([ones(size(X,1),1), X] * beta);
end

function y = sigmoid(x)
    x = max(min(x, 50), -50);
    y = 1 ./ (1 + exp(-x));
end

function bestThreshold = choose_best_threshold_f2(yTrue, score)
    yTrue = yTrue(:);
    score = score(:);

    thresholds = 0.01:0.01:0.99;
    bestF2 = -inf;
    bestThreshold = 0.5;

    for t = thresholds
        pred = double(score >= t);
        TP = sum(yTrue == 1 & pred == 1);
        FP = sum(yTrue == 0 & pred == 1);
        FN = sum(yTrue == 1 & pred == 0);

        precision = TP / max(TP + FP, 1);
        recall = TP / max(TP + FN, 1);
        F2 = 5 * precision * recall / max(4 * precision + recall, eps);

        if F2 > bestF2
            bestF2 = F2;
            bestThreshold = t;
        end
    end
end

function metrics = classification_metrics(yTrue, yPred, score)
    yTrue = yTrue(:);
    yPred = yPred(:);
    score = score(:);

    valid = ~isnan(yTrue) & ~isnan(yPred) & ~isnan(score) & ...
            isfinite(yTrue) & isfinite(yPred) & isfinite(score);

    yTrue = yTrue(valid);
    yPred = yPred(valid);
    score = score(valid);

    TP = sum(yTrue == 1 & yPred == 1);
    TN = sum(yTrue == 0 & yPred == 0);
    FP = sum(yTrue == 0 & yPred == 1);
    FN = sum(yTrue == 1 & yPred == 0);

    metrics = struct();
    metrics.Accuracy = (TP + TN) / max(TP + TN + FP + FN, 1);
    metrics.Precision = TP / max(TP + FP, 1);
    metrics.Recall = TP / max(TP + FN, 1);
    metrics.F1 = 2 * metrics.Precision * metrics.Recall / ...
        max(metrics.Precision + metrics.Recall, eps);
    metrics.F2 = 5 * metrics.Precision * metrics.Recall / ...
        max(4 * metrics.Precision + metrics.Recall, eps);
    metrics.AUC = manual_auc(yTrue, score);

    metrics.TP = TP;
    metrics.TN = TN;
    metrics.FP = FP;
    metrics.FN = FN;
end

function auc = manual_auc(yTrue, score)
    posScore = score(yTrue == 1);
    negScore = score(yTrue == 0);

    nPos = numel(posScore);
    nNeg = numel(negScore);

    if nPos == 0 || nNeg == 0
        auc = NaN;
        return;
    end

    countGreater = 0;
    countEqual = 0;

    for i = 1:nPos
        countGreater = countGreater + sum(posScore(i) > negScore);
        countEqual = countEqual + sum(posScore(i) == negScore);
    end

    auc = (countGreater + 0.5 * countEqual) / (nPos * nNeg);
end

function rapTable = rapping_peak_rate_table(T, varName, nGroup, peakVarName)
    valid = T.PeakValid & ~isnan(T.(varName)) & ~isnan(T.(peakVarName));
    x = T.(varName)(valid);
    y = T.(peakVarName)(valid);

    edges = quantile(x, linspace(0,1,nGroup+1));
    edges(1) = -inf;
    edges(end) = inf;

    group = discretize(x, edges);

    RappingParam = strings(nGroup,1);
    Group = (1:nGroup)';
    Mean_RappingPeriod_s = NaN(nGroup,1);
    SampleCount = zeros(nGroup,1);
    PeakRate = NaN(nGroup,1);

    for g = 1:nGroup
        idx = group == g;
        RappingParam(g) = string(varName);
        Mean_RappingPeriod_s(g) = mean(x(idx), "omitnan");
        SampleCount(g) = sum(idx);
        PeakRate(g) = mean(y(idx), "omitnan");
    end

    rapTable = table(RappingParam, Group, Mean_RappingPeriod_s, SampleCount, PeakRate);
end

function arr = pyvector_to_double(pyObj)
    try
        arr = double(py.array.array('d', py.numpy.nditer(pyObj)));
    catch
        try
            cellObj = cell(pyObj.tolist());
            arr = cellfun(@double, cellObj);
        catch
            arr = double(pyObj);
        end
    end
    arr = arr(:);
end

function score = catboost_predict_proba_positive(model, Xpy)
    proba = model.predict_proba(Xpy);
    probaCell = cell(proba.tolist());
    n = numel(probaCell);
    score = zeros(n,1);

    for i = 1:n
        row = cell(probaCell{i});
        if numel(row) >= 2
            score(i) = double(row{2});
        else
            score(i) = double(row{1});
        end
    end
end

function plot_prediction(yTrue, yPred, titleStr, xlab, ylab, savePath)
    figure('Color','w');
    plot(yTrue, 'k-', 'LineWidth', 1); hold on;
    plot(yPred, 'r-', 'LineWidth', 1);
    grid on;
    xlabel(xlab);
    ylabel(ylab);
    title(titleStr);
    legend("真实值", "预测值", "Location", "best");
    saveas(gcf, savePath);
    close(gcf);
end

function plot_scatter(yTrue, yPred, titleStr, xlab, ylab, savePath)
    figure('Color','w');
    scatter(yTrue, yPred, 18, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    mn = min([yTrue(:); yPred(:)]);
    mx = max([yTrue(:); yPred(:)]);
    plot([mn mx], [mn mx], 'r--', 'LineWidth', 1.2);
    grid on;
    xlabel(xlab);
    ylabel(ylab);
    title(titleStr);
    axis tight;
    saveas(gcf, savePath);
    close(gcf);
end

function plot_bar_importance(names, values, titleStr, savePath, topN)
    topN = min(topN, numel(values));
    names = names(1:topN);
    values = values(1:topN);

    figure('Color','w');
    barh(flipud(values));
    yticks(1:topN);
    yticklabels(flipud(names));
    xlabel("重要性");
    title(titleStr);
    grid on;
    saveas(gcf, savePath);
    close(gcf);
end

function plot_peak_detection(t, cout, yTrue, yPred, score, savePath)
    figure('Color','w');

    subplot(2,1,1);
    plot(t, cout, 'k-', 'LineWidth', 1); hold on;
    scatter(t(yTrue==1), cout(yTrue==1), 28, 'r', 'filled');
    scatter(t(yPred==1), cout(yPred==1), 35, 'bo');
    grid on;
    title("峰值点识别结果");
    ylabel("C_{out}");
    legend("C_{out}", "真实峰值点", "预测峰值点", "Location", "best");

    subplot(2,1,2);
    plot(t, score, 'b-', 'LineWidth', 1);
    grid on;
    ylabel("峰值概率");
    xlabel("时间");
    title("模型输出峰值概率");

    saveas(gcf, savePath);
    close(gcf);
end

function plot_rapping_peak_rate(allRapTables, savePath)
    figure('Color','w');
    vars = unique(allRapTables.RappingParam, 'stable');

    for i = 1:numel(vars)
        subplot(2,2,i);
        idx = allRapTables.RappingParam == vars(i);
        plot(allRapTables.Mean_RappingPeriod_s(idx), ...
             100 * allRapTables.PeakRate(idx), ...
             'o-', 'LineWidth', 1.5);
        grid on;
        xlabel("平均振打周期 / s");
        ylabel("峰值事件率 / %");
        title(string(vars(i)) + " 分组峰值事件率");
    end

    saveas(gcf, savePath);
    close(gcf);
end