%% run_problem1.m
% ------------------------------------------------------------
% 问题一：
% 分析入口条件、操作参数与出口粉尘浓度的关系，
% 并研究振打周期对瞬时排放峰值的影响。
%
% 方法：
%   1. 构造综合除尘强度 K
%   2. 构造机理特征和动态滞后特征
%   3. MATLAB 调用 Python CatBoostRegressor 预测 K
%   4. 由 K 反推出出口粉尘浓度 C_out
%   5. 构造瞬时排放峰值标签
%   6. MATLAB 调用 Python CatBoostClassifier 预测峰值风险
%   7. 输出评价指标、特征重要性、振打周期峰值率和图表
% ------------------------------------------------------------

clear; clc; close all;

%% ============================================================
% 0. 工程路径初始化
%% ============================================================

scriptPath = mfilename('fullpath');
scriptDir = fileparts(scriptPath);
projectRoot = fileparts(scriptDir);

cd(projectRoot);
addpath(genpath(projectRoot));

dataPath = fullfile(projectRoot, "00_data", "raw", "Cement_ESP_Data.csv");

resultDir = fullfile(projectRoot, "04_results", "problem1");
figDir = fullfile(resultDir, "figures");
tableDir = fullfile(resultDir, "tables");
modelDir = fullfile(resultDir, "models");
logDir = fullfile(resultDir, "logs");
processedDir = fullfile(projectRoot, "00_data", "processed");

make_dir_if_not_exist(resultDir);
make_dir_if_not_exist(figDir);
make_dir_if_not_exist(tableDir);
make_dir_if_not_exist(modelDir);
make_dir_if_not_exist(logDir);
make_dir_if_not_exist(processedDir);

rng(2026);

print_title("问题一：入口条件、操作参数与出口浓度关系分析");

%% ============================================================
% 1. 检查 Python 和 CatBoost
%% ============================================================

print_title("1. 检查 Python 与 CatBoost 环境");

disp("当前 MATLAB 使用的 Python 环境：");
disp(pyenv);

try
    catboost = py.importlib.import_module('catboost');
    np = py.importlib.import_module('numpy');
    fprintf("[成功] 已成功导入 Python catboost 和 numpy。\n");
catch ME
    fprintf("[失败] 无法导入 catboost 或 numpy。\n");
    fprintf("请先在命令行运行：pip install catboost numpy scikit-learn\n");
    rethrow(ME);
end

%% ============================================================
% 2. 读取数据
%% ============================================================

print_title("2. 读取原始数据");

if ~isfile(dataPath)
    error("找不到数据文件：%s", dataPath);
end

opts = detectImportOptions(dataPath, "VariableNamingRule", "preserve");
T = readtable(dataPath, opts);

fprintf("原始样本数：%d\n", height(T));
disp("字段名：");
disp(T.Properties.VariableNames');

% 时间戳转换
if ~isdatetime(T.timestamp)
    try
        T.timestamp = datetime(T.timestamp, "InputFormat", "yyyy-MM-dd HH:mm:ss");
    catch
        T.timestamp = datetime(string(T.timestamp));
    end
end

T = sortrows(T, "timestamp");

%% ============================================================
% 3. 数据清洗
%% ============================================================

print_title("3. 数据清洗");

numericCols = {
    'Temp_C','C_in_gNm3','Q_Nm3h', ...
    'U1_kV','U2_kV','U3_kV','U4_kV', ...
    'T1_s','T2_s','T3_s','T4_s', ...
    'C_out_mgNm3','P_total_kW'
};

for i = 1:numel(numericCols)
    col = numericCols{i};
    if ~isnumeric(T.(col))
        T.(col) = str2double(string(T.(col)));
    end
end

% 输入变量必须完整
inputCols = {
    'Temp_C','C_in_gNm3','Q_Nm3h', ...
    'U1_kV','U2_kV','U3_kV','U4_kV', ...
    'T1_s','T2_s','T3_s','T4_s'
};

T = rmmissing(T, "DataVariables", inputCols);

% 基础物理合理性过滤
valid = true(height(T),1);
valid = valid & T.Temp_C > 0 & T.Temp_C < 300;
valid = valid & T.C_in_gNm3 > 0 & T.C_in_gNm3 < 500;
valid = valid & T.Q_Nm3h > 0;
valid = valid & T.U1_kV > 0 & T.U2_kV > 0 & T.U3_kV > 0 & T.U4_kV > 0;
valid = valid & T.T1_s > 0 & T.T2_s > 0 & T.T3_s > 0 & T.T4_s > 0;

T = T(valid,:);

% 出口浓度和电耗不在这里强行删除，只把非正数设为缺失
badCout = ~isnan(T.C_out_mgNm3) & T.C_out_mgNm3 <= 0;
T.C_out_mgNm3(badCout) = NaN;

badP = ~isnan(T.P_total_kW) & T.P_total_kW <= 0;
T.P_total_kW(badP) = NaN;

% 时间去重
[~, ia] = unique(T.timestamp, 'stable');
T = T(ia,:);
T = sortrows(T, "timestamp");

fprintf("清洗后样本数：%d\n", height(T));
fprintf("有效出口浓度样本数：%d\n", sum(~isnan(T.C_out_mgNm3)));
fprintf("出口浓度缺失样本数：%d\n", sum(isnan(T.C_out_mgNm3)));

save(fullfile(processedDir, "cleaned_data.mat"), "T");

%% ============================================================
% 4. 构造机理特征
%% ============================================================

print_title("4. 构造机理特征");

% 入口浓度从 g/Nm3 转为 mg/Nm3
T.C_in_mgNm3 = 1000 * T.C_in_gNm3;

% 有效出口浓度标记
T.HasCout = ~isnan(T.C_out_mgNm3) & T.C_out_mgNm3 > 0 & T.C_in_mgNm3 > 0;

% 综合除尘效率
T.eta = NaN(height(T),1);
T.eta(T.HasCout) = 1 - T.C_out_mgNm3(T.HasCout) ./ T.C_in_mgNm3(T.HasCout);

% 综合除尘强度 K
% K = -ln(C_out / (1000*C_in))
T.K = NaN(height(T),1);
T.K(T.HasCout) = -log(T.C_out_mgNm3(T.HasCout) ./ T.C_in_mgNm3(T.HasCout));

T.ValidK = T.HasCout & isfinite(T.K) & T.K > 0;

fprintf("有效 K 样本数：%d\n", sum(T.ValidK));

% 粉尘负荷
T.DustLoad = T.C_in_gNm3 .* T.Q_Nm3h;

% 电压组合特征
T.U_sum = T.U1_kV + T.U2_kV + T.U3_kV + T.U4_kV;
T.U_mean = T.U_sum / 4;
T.U_sqsum = T.U1_kV.^2 + T.U2_kV.^2 + T.U3_kV.^2 + T.U4_kV.^2;

T.U_front = T.U1_kV + T.U2_kV;
T.U_back  = T.U3_kV + T.U4_kV;
T.U_ratio_front_back = T.U_front ./ max(T.U_back, eps);

% 振打周期组合特征
T.T_mean = (T.T1_s + T.T2_s + T.T3_s + T.T4_s) / 4;
T.T_front = T.T1_s + T.T2_s;
T.T_back  = T.T3_s + T.T4_s;
T.T_diff_back_front = T.T_back - T.T_front;

% 单位电场强度承载的粉尘负荷，可理解为除尘压力
T.LoadPressure = T.DustLoad ./ max(T.U_sqsum, eps);

%% ============================================================
% 5. 构造动态滞后特征
%% ============================================================

print_title("5. 构造动态滞后特征");

lagList = [1, 3, 5, 10, 15, 30];

% 出口浓度滞后
for lag = lagList
    T.(sprintf("Cout_lag%d", lag)) = lag_vector(T.C_out_mgNm3, lag);
end

% K 滞后
for lag = lagList
    T.(sprintf("K_lag%d", lag)) = lag_vector(T.K, lag);
end

% 电压变化量
for i = 1:4
    uname = sprintf("U%d_kV", i);
    T.(sprintf("dU%d", i)) = [NaN; diff(T.(uname))];
end

% 振打周期变化量
for i = 1:4
    tname = sprintf("T%d_s", i);
    T.(sprintf("dT%d", i)) = [NaN; diff(T.(tname))];
end

% 过去 30 分钟滚动统计，先滞后一位，避免使用当前值造成信息泄漏
window = 30;

CoutLag1 = lag_vector(T.C_out_mgNm3, 1);
T.Cout_roll_mean30 = movmean(CoutLag1, [window-1 0], "omitnan");
T.Cout_roll_std30  = movstd(CoutLag1, [window-1 0], "omitnan");

KLag1 = lag_vector(T.K, 1);
T.K_roll_mean30 = movmean(KLag1, [window-1 0], "omitnan");
T.K_roll_std30  = movstd(KLag1, [window-1 0], "omitnan");

%% ============================================================
% 6. 构造瞬时排放峰值标签
%% ============================================================

print_title("6. 构造瞬时排放峰值标签");

peakSigmaCoef = 2;

T.PeakValid = ~isnan(T.C_out_mgNm3) & ...
              ~isnan(T.Cout_roll_mean30) & ...
              ~isnan(T.Cout_roll_std30);

T.Peak = false(height(T),1);

T.Peak(T.PeakValid) = T.C_out_mgNm3(T.PeakValid) > ...
    T.Cout_roll_mean30(T.PeakValid) + peakSigmaCoef * T.Cout_roll_std30(T.PeakValid);

fprintf("峰值可判定样本数：%d\n", sum(T.PeakValid));
fprintf("识别到峰值样本数：%d\n", sum(T.Peak));
fprintf("峰值样本占比：%.4f%%\n", 100 * mean(T.Peak(T.PeakValid)));

save(fullfile(processedDir, "feature_data.mat"), "T");

%% ============================================================
% 7. 定义模型特征
%% ============================================================

featureNames = {
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

%% ============================================================
% 8. 构造 K 回归数据集
%% ============================================================

print_title("7. 构造 K 回归数据集");

regData = T(T.ValidK,:);

% 注意：
% CatBoost 可以处理 NaN，因此这里不要对全部特征 rmmissing，
% 否则会因为滞后特征缺失损失大量样本。
regData = rmmissing(regData, "DataVariables", {'K','C_out_mgNm3','C_in_mgNm3'});

X = table2array(regData(:, featureNames));
yK = regData.K;
yCout = regData.C_out_mgNm3;
CinTestAll = regData.C_in_mgNm3;

n = size(X,1);
if n < 100
    warning("有效回归样本数较少：%d，模型结果可能不稳定。", n);
end

[idxTrain, idxVal, idxTest] = time_split_indices(n, 0.70, 0.15, 0.15);

XTrain = X(idxTrain,:);
yTrain = yK(idxTrain);

XVal = X(idxVal,:);
yVal = yK(idxVal);

XTest = X(idxTest,:);
yTestK = yK(idxTest);
yTestCout = yCout(idxTest);
CinTest = CinTestAll(idxTest);

fprintf("回归训练集样本数：%d\n", numel(idxTrain));
fprintf("回归验证集样本数：%d\n", numel(idxVal));
fprintf("回归测试集样本数：%d\n", numel(idxTest));

problem1_regression_data = struct();
problem1_regression_data.featureNames = featureNames;
problem1_regression_data.regData = regData;
problem1_regression_data.idxTrain = idxTrain;
problem1_regression_data.idxVal = idxVal;
problem1_regression_data.idxTest = idxTest;
save(fullfile(processedDir, "problem1_regression_data.mat"), "problem1_regression_data");

%% ============================================================
% 9. CatBoost 回归模型：预测 K
%% ============================================================

print_title("8. 训练 CatBoostRegressor 预测综合除尘强度 K");

modelK = catboost.CatBoostRegressor(pyargs( ...
    'iterations', int32(2000), ...
    'depth', int32(6), ...
    'learning_rate', 0.03, ...
    'loss_function', 'RMSE', ...
    'eval_metric', 'RMSE', ...
    'random_seed', int32(2026), ...
    'verbose', false ...
));

XTrainPy = np.array(XTrain);
yTrainPy = np.array(yTrain(:));
XValPy = np.array(XVal);
yValPy = np.array(yVal(:));

evalSet = py.tuple({XValPy, yValPy});

modelK.fit(XTrainPy, yTrainPy, pyargs( ...
    'eval_set', evalSet, ...
    'use_best_model', true, ...
    'early_stopping_rounds', int32(100) ...
));

fprintf("CatBoostRegressor 训练完成。\n");

KPredRaw = pyvector_to_double(modelK.predict(np.array(XTest)));
KPred = max(KPredRaw(:), 0);

% 由 K 反推出口浓度
CoutPred = CinTest(:) .* exp(-KPred(:));

%% ============================================================
% 10. 回归模型评估
%% ============================================================

print_title("9. 回归模型评估");

metricsK = regression_metrics(yTestK, KPred);
metricsCout = regression_metrics(yTestCout, CoutPred);

fprintf("\n================ K 预测评估结果 ================\n");
disp(struct2table(metricsK));

fprintf("\n============= 出口浓度 C_out 反推评估结果 =============\n");
disp(struct2table(metricsCout));

writetable(struct2table(metricsK), fullfile(tableDir, "P1_K_Metrics.csv"));
writetable(struct2table(metricsCout), fullfile(tableDir, "P1_Cout_Metrics.csv"));

% 保存模型
try
    modelK.save_model(char(fullfile(modelDir, "P1_CatBoost_K_Model.cbm")));
    fprintf("K 回归模型已保存。\n");
catch
    warning("K 回归模型保存失败。");
end

%% ============================================================
% 11. K 回归模型特征重要性
%% ============================================================

print_title("10. K 回归模型特征重要性");

impK = get_catboost_importance(modelK, featureNames);
disp(impK(1:min(20,height(impK)),:));

writetable(impK, fullfile(tableDir, "P1_K_FeatureImportance.csv"));

%% ============================================================
% 12. 绘制回归结果图
%% ============================================================

plot_prediction(yTestCout, CoutPred, ...
    "出口粉尘浓度真实值与预测值对比", ...
    "样本序号", "C_{out} / mg·Nm^{-3}", ...
    fullfile(figDir, "P1_Cout_Prediction.png"));

plot_prediction(yTestK, KPred, ...
    "综合除尘强度 K 真实值与预测值对比", ...
    "样本序号", "K", ...
    fullfile(figDir, "P1_K_Prediction.png"));

plot_scatter(yTestK, KPred, ...
    "综合除尘强度 K 真实值-预测值散点图", ...
    "真实 K", "预测 K", ...
    fullfile(figDir, "P1_K_Scatter.png"));

plot_feature_importance(impK, 20, ...
    "K 回归模型特征重要性 Top 20", ...
    fullfile(figDir, "P1_K_FeatureImportance.png"));

%% ============================================================
% 13. 构造峰值分类数据集
%% ============================================================

print_title("11. 构造峰值分类数据集");

clsData = T(T.PeakValid & T.ValidK,:);

Xc = table2array(clsData(:, featureNames));
yc = double(clsData.Peak);

nCls = size(Xc,1);

[idxTrainC, idxValC, idxTestC] = time_split_indices(nCls, 0.70, 0.15, 0.15);

XcTrain = Xc(idxTrainC,:);
ycTrain = yc(idxTrainC);

XcVal = Xc(idxValC,:);
ycVal = yc(idxValC);

XcTest = Xc(idxTestC,:);
ycTest = yc(idxTestC);

fprintf("分类训练集样本数：%d，峰值比例：%.4f%%\n", numel(idxTrainC), 100*mean(ycTrain));
fprintf("分类验证集样本数：%d，峰值比例：%.4f%%\n", numel(idxValC), 100*mean(ycVal));
fprintf("分类测试集样本数：%d，峰值比例：%.4f%%\n", numel(idxTestC), 100*mean(ycTest));

if numel(unique(ycTrain)) < 2
    warning("训练集中只有一个类别，无法训练峰值分类模型。请检查峰值定义或数据分布。");
else

    problem1_peak_data = struct();
    problem1_peak_data.featureNames = featureNames;
    problem1_peak_data.clsData = clsData;
    problem1_peak_data.idxTrain = idxTrainC;
    problem1_peak_data.idxVal = idxValC;
    problem1_peak_data.idxTest = idxTestC;
    save(fullfile(processedDir, "problem1_peak_data.mat"), "problem1_peak_data");

    %% ========================================================
    % 14. CatBoost 峰值分类模型
    %% ========================================================

    print_title("12. 训练 CatBoostClassifier 识别瞬时排放峰值");

    modelPeak = catboost.CatBoostClassifier(pyargs( ...
        'iterations', int32(1500), ...
        'depth', int32(5), ...
        'learning_rate', 0.03, ...
        'loss_function', 'Logloss', ...
        'eval_metric', 'AUC', ...
        'auto_class_weights', 'Balanced', ...
        'random_seed', int32(2026), ...
        'verbose', false ...
    ));

    XcTrainPy = np.array(XcTrain);
    ycTrainPy = np.array(ycTrain(:));
    XcValPy = np.array(XcVal);
    ycValPy = np.array(ycVal(:));

    evalSetCls = py.tuple({XcValPy, ycValPy});

    modelPeak.fit(XcTrainPy, ycTrainPy, pyargs( ...
        'eval_set', evalSetCls, ...
        'use_best_model', true, ...
        'early_stopping_rounds', int32(100) ...
    ));

    fprintf("CatBoostClassifier 训练完成。\n");

    %% ========================================================
    % 15. 阈值选择与分类评估
    %% ========================================================

    print_title("13. 峰值分类模型评估");

    probVal = pymatrix_to_double(modelPeak.predict_proba(np.array(XcVal)));
    scoreVal = probVal(:,2);

    bestThreshold = choose_best_threshold(ycVal, scoreVal);

    probTest = pymatrix_to_double(modelPeak.predict_proba(np.array(XcTest)));
    scoreTest = probTest(:,2);

    peakPred = double(scoreTest >= bestThreshold);

    metricsPeak = classification_metrics(ycTest, peakPred, scoreTest);

    fprintf("最优分类阈值：%.4f\n", bestThreshold);
    fprintf("\n================ 峰值分类模型评估结果 ================\n");
    disp(struct2table(metricsPeak));

    writetable(struct2table(metricsPeak), fullfile(tableDir, "P1_Peak_Metrics.csv"));

    try
        modelPeak.save_model(char(fullfile(modelDir, "P1_CatBoost_Peak_Model.cbm")));
        fprintf("峰值分类模型已保存。\n");
    catch
        warning("峰值分类模型保存失败。");
    end

    %% ========================================================
    % 16. 峰值模型特征重要性
    %% ========================================================

    print_title("14. 峰值分类模型特征重要性");

    impPeak = get_catboost_importance(modelPeak, featureNames);
    disp(impPeak(1:min(20,height(impPeak)),:));

    writetable(impPeak, fullfile(tableDir, "P1_Peak_FeatureImportance.csv"));

    plot_feature_importance(impPeak, 20, ...
        "瞬时排放峰值分类模型特征重要性 Top 20", ...
        fullfile(figDir, "P1_Peak_FeatureImportance.png"));
end

%% ============================================================
% 17. 出口浓度峰值识别图
%% ============================================================

print_title("15. 绘制峰值识别图");

figure('Color','w');
plot(T.timestamp, T.C_out_mgNm3, 'b-', 'LineWidth', 1.0); hold on;
scatter(T.timestamp(T.Peak), T.C_out_mgNm3(T.Peak), 25, 'r', 'filled');
grid on;
xlabel('时间');
ylabel('C_{out} / mg·Nm^{-3}');
legend('出口浓度','识别峰值','Location','best');
title('出口粉尘浓度时间序列及瞬时排放峰值识别');
saveas(gcf, fullfile(figDir, "P1_Peak_Detection.png"));

%% ============================================================
% 18. 振打周期与峰值频率关系分析
%% ============================================================

print_title("16. 振打周期分组峰值频率分析");

rappingCols = {'T1_s','T2_s','T3_s','T4_s'};
allPeakRateTables = table();

for i = 1:numel(rappingCols)
    col = rappingCols{i};

    validGroup = ~isnan(T.(col)) & T.PeakValid;
    x = T.(col)(validGroup);
    y = double(T.Peak(validGroup));

    edges = quantile(x, [0 0.2 0.4 0.6 0.8 1.0]);
    edges = unique(edges);

    if numel(edges) < 3
        fprintf("%s 分组边界不足，跳过。\n", col);
        continue;
    end

    bin = discretize(x, edges);
    groupID = unique(bin(~isnan(bin)));

    paramName = strings(numel(groupID),1);
    meanPeriod = zeros(numel(groupID),1);
    sampleCount = zeros(numel(groupID),1);
    peakRate = zeros(numel(groupID),1);

    for j = 1:numel(groupID)
        id = groupID(j);
        idx = bin == id;

        paramName(j) = string(col);
        meanPeriod(j) = mean(x(idx), "omitnan");
        sampleCount(j) = sum(idx);
        peakRate(j) = mean(y(idx), "omitnan");
    end

    resultTable = table(paramName, groupID(:), meanPeriod(:), sampleCount(:), peakRate(:), ...
        'VariableNames', {'RappingParam','Group','Mean_RappingPeriod_s','SampleCount','PeakRate'});

    fprintf("\n%s 振打周期分组峰值率：\n", col);
    disp(resultTable);

    allPeakRateTables = [allPeakRateTables; resultTable]; %#ok<AGROW>

    writetable(resultTable, fullfile(tableDir, sprintf("P1_%s_PeakRate.csv", col)));
end

writetable(allPeakRateTables, fullfile(tableDir, "P1_All_RappingPeriod_PeakRate.csv"));

%% ============================================================
% 19. 保存汇总结果
%% ============================================================

print_title("17. 保存问题一汇总结果");

problem1_results = struct();
problem1_results.metricsK = metricsK;
problem1_results.metricsCout = metricsCout;
problem1_results.impK = impK;
problem1_results.allPeakRateTables = allPeakRateTables;

if exist("metricsPeak", "var")
    problem1_results.metricsPeak = metricsPeak;
end

if exist("impPeak", "var")
    problem1_results.impPeak = impPeak;
end

save(fullfile(resultDir, "P1_results_summary.mat"), "problem1_results");

fprintf("\n问题一运行完成。\n");
fprintf("结果保存目录：%s\n", resultDir);

fprintf("\n主要输出文件：\n");
fprintf("1. %s\n", fullfile(tableDir, "P1_K_Metrics.csv"));
fprintf("2. %s\n", fullfile(tableDir, "P1_Cout_Metrics.csv"));
fprintf("3. %s\n", fullfile(tableDir, "P1_K_FeatureImportance.csv"));
fprintf("4. %s\n", fullfile(tableDir, "P1_All_RappingPeriod_PeakRate.csv"));
fprintf("5. %s\n", fullfile(figDir, "P1_Cout_Prediction.png"));
fprintf("6. %s\n", fullfile(figDir, "P1_Peak_Detection.png"));

%% ============================================================
% 局部函数
%% ============================================================

function print_title(titleText)
    fprintf("\n");
    fprintf("============================================================\n");
    fprintf("%s\n", titleText);
    fprintf("============================================================\n");
end

function make_dir_if_not_exist(dirPath)
    if ~exist(dirPath, 'dir')
        mkdir(dirPath);
    end
end

function y = lag_vector(x, lag)
    y = NaN(size(x));
    if lag < numel(x)
        y(lag+1:end) = x(1:end-lag);
    end
end

function [idxTrain, idxVal, idxTest] = time_split_indices(n, trainRatio, valRatio, testRatio)
    if abs(trainRatio + valRatio + testRatio - 1) > 1e-8
        error("训练、验证、测试比例之和必须为 1。");
    end

    idxTrainEnd = floor(trainRatio * n);
    idxValEnd = floor((trainRatio + valRatio) * n);

    idxTrain = 1:idxTrainEnd;
    idxVal = idxTrainEnd+1:idxValEnd;
    idxTest = idxValEnd+1:n;
end

function v = pyvector_to_double(pyObj)
    try
        pyObj = pyObj.tolist();
    catch
    end

    c = cell(pyObj);
    v = zeros(numel(c),1);

    for i = 1:numel(c)
        item = c{i};

        if isa(item, 'py.list') || isa(item, 'py.tuple')
            temp = cell(item);
            v(i) = double(temp{1});
        else
            v(i) = double(item);
        end
    end
end

function M = pymatrix_to_double(pyObj)
    try
        pyObj = pyObj.tolist();
    catch
    end

    rows = cell(pyObj);
    n = numel(rows);

    firstRow = cell(rows{1});
    m = numel(firstRow);

    M = zeros(n,m);

    for i = 1:n
        row = cell(rows{i});
        for j = 1:m
            M(i,j) = double(row{j});
        end
    end
end

function metrics = regression_metrics(yTrue, yPred)
    yTrue = yTrue(:);
    yPred = yPred(:);

    err = yTrue - yPred;

    metrics = struct();
    metrics.MAE = mean(abs(err), "omitnan");
    metrics.RMSE = sqrt(mean(err.^2, "omitnan"));

    ssRes = sum(err.^2, "omitnan");
    ssTot = sum((yTrue - mean(yTrue,"omitnan")).^2, "omitnan");
    metrics.R2 = 1 - ssRes / max(ssTot, eps);

    valid = abs(yTrue) > eps;
    metrics.MAPE = mean(abs(err(valid) ./ yTrue(valid)), "omitnan") * 100;
end

function metrics = classification_metrics(yTrue, yPred, score)
    yTrue = yTrue(:);
    yPred = yPred(:);
    score = score(:);

    TP = sum(yTrue == 1 & yPred == 1);
    TN = sum(yTrue == 0 & yPred == 0);
    FP = sum(yTrue == 0 & yPred == 1);
    FN = sum(yTrue == 1 & yPred == 0);

    metrics = struct();
    metrics.Accuracy = (TP + TN) / max(TP + TN + FP + FN, 1);
    metrics.Precision = TP / max(TP + FP, 1);
    metrics.Recall = TP / max(TP + FN, 1);
    metrics.F1 = 2 * metrics.Precision * metrics.Recall / max(metrics.Precision + metrics.Recall, eps);

    try
        [~,~,~,auc] = perfcurve(yTrue, score, 1);
        metrics.AUC = auc;
    catch
        metrics.AUC = NaN;
    end

    metrics.TP = TP;
    metrics.TN = TN;
    metrics.FP = FP;
    metrics.FN = FN;
end

function bestThreshold = choose_best_threshold(yTrue, score)
    yTrue = yTrue(:);
    score = score(:);

    thresholds = 0.05:0.01:0.95;
    bestF1 = -inf;
    bestThreshold = 0.5;

    for t = thresholds
        pred = double(score >= t);
        m = classification_metrics(yTrue, pred, score);

        if m.F1 > bestF1
            bestF1 = m.F1;
            bestThreshold = t;
        end
    end
end

function impTable = get_catboost_importance(model, featureNames)
    impPy = model.get_feature_importance(pyargs('type','PredictionValuesChange'));
    imp = pyvector_to_double(impPy);

    impTable = table(featureNames(:), imp(:), ...
        'VariableNames', {'Feature','Importance'});

    impTable = sortrows(impTable, "Importance", "descend");
end

function plot_prediction(yTrue, yPred, figTitle, xText, yText, savePath)
    figure('Color','w');
    plot(yTrue, 'b-', 'LineWidth', 1.2); hold on;
    plot(yPred, 'r--', 'LineWidth', 1.2);
    grid on;
    xlabel(xText);
    ylabel(yText);
    legend('真实值','预测值','Location','best');
    title(figTitle);
    saveas(gcf, savePath);
end

function plot_scatter(yTrue, yPred, figTitle, xText, yText, savePath)
    figure('Color','w');
    scatter(yTrue, yPred, 20, 'filled'); hold on;
    grid on;

    minVal = min([yTrue(:); yPred(:)]);
    maxVal = max([yTrue(:); yPred(:)]);
    plot([minVal maxVal], [minVal maxVal], 'r--', 'LineWidth', 1.2);

    xlabel(xText);
    ylabel(yText);
    title(figTitle);
    saveas(gcf, savePath);
end

function plot_feature_importance(impTable, topN, figTitle, savePath)
    topN = min(topN, height(impTable));
    subTable = impTable(1:topN,:);

    figure('Color','w');
    barh(flipud(subTable.Importance));
    yticks(1:topN);
    yticklabels(flipud(subTable.Feature));
    xlabel('Importance');
    title(figTitle);
    grid on;
    saveas(gcf, savePath);
end