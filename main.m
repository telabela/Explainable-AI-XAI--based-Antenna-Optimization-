
clc; clear; close all;

%% Step 1: Generate Synthetic Antenna Data with FR4 Substrate
numSamples = 5000; % Number of data points
freq = 15e9; % 15 GHz frequency
c = 3e8; % Speed of light (m/s)
lambda = c / freq; % Wavelength

% FR4 Substrate Properties
epsilon_r = 4.4;  % Relative Permittivity
loss_tangent = 0.02;

% Define input antenna dimensions (Width, Length, Height)
Width = rand(numSamples,1) * lambda * 1.5;
Length = rand(numSamples,1) * lambda * 1.5;
Height = rand(numSamples,1) * lambda * 0.5;

% Compute Outputs Using Formulas
Efficiency = 0.9; % Assumed Antenna Efficiency
Gain = Efficiency * (4 * pi * (Width .* Length) ./ lambda.^2);  % Gain Formula
Bandwidth = (c ./ (2 * pi * sqrt(epsilon_r))) .* (1 ./ Length);  % Bandwidth Formula
Z_in = 50 + 5 * randn(numSamples,1);  % Simulated Impedance
S11 = -20 * log10(abs((Z_in - 50) ./ (Z_in + 50)));  % S11 Formula

% Combine into dataset
data = table(Width, Length, Height, Z_in, Gain, Bandwidth, S11);

%% Step 2: Visualizing Heatmap of Feature Correlation
figure;
corrMatrix = corr(table2array(data));
heatmap({'Width', 'Length', 'Height', 'Z_in', 'Gain', 'Bandwidth', 'S11'}, ...
        {'Width', 'Length', 'Height', 'Z_in', 'Gain', 'Bandwidth', 'S11'}, ...
        corrMatrix, 'Colormap', parula, 'Title', 'Feature Correlation Heatmap');

%% Step 3: Split Data into Training & Testing
X = table2array(data(:,1:4)); % Inputs (Width, Length, Height, Z_in)
Y = table2array(data(:,5:7)); % Outputs (Gain, Bandwidth, S11)
[X_train, X_test, Y_train, Y_test] = deal(X(1:4000,:), X(4001:end,:), Y(1:4000,:), Y(4001:end,:));

%% Step 4: Train Decision Tree Model (Best Performing)
mdl_tree_gain = fitrtree(X_train, Y_train(:,1));  % Decision Tree for Gain
mdl_tree_bw = fitrtree(X_train, Y_train(:,2));    % Decision Tree for Bandwidth
mdl_tree_s11 = fitrtree(X_train, Y_train(:,3));   % Decision Tree for S11

% Predict outputs for test data
Y_pred_gain = predict(mdl_tree_gain, X_test);
Y_pred_bw = predict(mdl_tree_bw, X_test);
Y_pred_s11 = predict(mdl_tree_s11, X_test);

%% Step 5: Regression Performance Metrics
metrics = @(y_true, y_pred) struct(...
    'RMSE', sqrt(mean((y_pred - y_true).^2)), ...
    'MAE', mean(abs(y_pred - y_true)), ...
    'R2', 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2));

perf_gain = metrics(Y_test(:,1), Y_pred_gain);
perf_bw = metrics(Y_test(:,2), Y_pred_bw);
perf_s11 = metrics(Y_test(:,3), Y_pred_s11);

fprintf('Performance Metrics for Gain Prediction:\n RMSE: %.4f, MAE: %.4f, R2: %.4f\n', ...
    perf_gain.RMSE, perf_gain.MAE, perf_gain.R2);
fprintf('Performance Metrics for Bandwidth Prediction:\n RMSE: %.4f, MAE: %.4f, R2: %.4f\n', ...
    perf_bw.RMSE, perf_bw.MAE, perf_bw.R2);
fprintf('Performance Metrics for S11 Prediction:\n RMSE: %.4f, MAE: %.4f, R2: %.4f\n', ...
    perf_s11.RMSE, perf_s11.MAE, perf_s11.R2);

%% Step 6: Regression Plots (Actual vs Predicted)
figure;
subplot(1,3,1);
scatter(Y_test(:,1), Y_pred_gain, 'filled'); hold on;
plot(Y_test(:,1), Y_test(:,1), 'r--', 'LineWidth', 1.5); % Perfect Fit Line
xlabel('Actual Gain'); ylabel('Predicted Gain'); title('Regression Plot: Gain');
grid on; axis equal;

subplot(1,3,2);
scatter(Y_test(:,2), Y_pred_bw, 'filled'); hold on;
plot(Y_test(:,2), Y_test(:,2), 'r--', 'LineWidth', 1.5);
xlabel('Actual Bandwidth'); ylabel('Predicted Bandwidth'); title('Regression Plot: Bandwidth');
grid on; axis equal;

subplot(1,3,3);
scatter(Y_test(:,3), Y_pred_s11, 'filled'); hold on;
plot(Y_test(:,3), Y_test(:,3), 'r--', 'LineWidth', 1.5);
xlabel('Actual S11'); ylabel('Predicted S11'); title('Regression Plot: S11');
grid on; axis equal;


%% Step 7: Explainable AI Plots

% Feature Importance for Decision Trees
figure;
subplot(1,3,1);
importance_gain = mdl_tree_gain.predictorImportance;
bar(importance_gain);
xlabel('Features'); ylabel('Importance'); title('Feature Importance: Gain');
xticklabels({'Width', 'Length', 'Height', 'Z_in'});

subplot(1,3,2);
importance_bw = mdl_tree_bw.predictorImportance;
bar(importance_bw);
xlabel('Features'); ylabel('Importance'); title('Feature Importance: Bandwidth');
xticklabels({'Width', 'Length', 'Height', 'Z_in'});

subplot(1,3,3);
importance_s11 = mdl_tree_s11.predictorImportance;
bar(importance_s11);
xlabel('Features'); ylabel('Importance'); title('Feature Importance: S11');
xticklabels({'Width', 'Length', 'Height', 'Z_in'});

% Partial Dependence Plots (PDPs)
figure;
subplot(1,3,1);
plotPartialDependence(mdl_tree_gain, 1); % PDP for Width vs Gain
xlabel('Width'); ylabel('Gain'); title('PDP: Width vs Gain');

subplot(1,3,2);
plotPartialDependence(mdl_tree_bw, 2); % PDP for Length vs Bandwidth
xlabel('Length'); ylabel('Bandwidth'); title('PDP: Length vs Bandwidth');

subplot(1,3,3);
plotPartialDependence(mdl_tree_s11, 3); % PDP for Height vs S11
xlabel('Height'); ylabel('S11'); title('PDP: Height vs S11');



%%
% Compute SHAP-like values for each model
shap_gain = compute_shap_values(mdl_tree_gain, X_train, 100); % Use 100 samples for estimation
shap_bw = compute_shap_values(mdl_tree_bw, X_train, 100);
shap_s11 = compute_shap_values(mdl_tree_s11, X_train, 100);
% Plot SHAP Bar Plot for Gain
figure;
subplot(1,3,1);
bar(shap_gain);
xlabel('Features'); ylabel('SHAP Value (Importance)'); title('SHAP Bar Plot: Gain');
xticklabels({'Width', 'Length', 'Height', 'Z_in'});

% Plot SHAP Bar Plot for Bandwidth
subplot(1,3,2);
bar(shap_bw);
xlabel('Features'); ylabel('SHAP Value (Importance)'); title('SHAP Bar Plot: Bandwidth');
xticklabels({'Width', 'Length', 'Height', 'Z_in'});

% Plot SHAP Bar Plot for S11
subplot(1,3,3);
bar(shap_s11);
xlabel('Features'); ylabel('SHAP Value (Importance)'); title('SHAP Bar Plot: S11');
xticklabels({'Width', 'Length', 'Height', 'Z_in'});

%%

import shap
from sklearn.tree import DecisionTreeRegressor

# Train a decision tree model (example)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Plot SHAP bar plot
shap.summary_plot(shap_values, X_train, plot_type="bar")








