function shap_values = compute_shap_values(model, X, num_samples)
    % model: Trained decision tree model
    % X: Input data (features)
    % num_samples: Number of samples to use for Shapley value estimation

    [n_samples, n_features] = size(X);
    shap_values = zeros(n_samples, n_features);

    for i = 1:n_samples
        % Select a random sample
        sample = X(i, :);

        % Create a background dataset (e.g., mean of all samples)
        background = mean(X, 1);

        % Compute Shapley values for each feature
        for j = 1:n_features
            % Create a modified sample with the j-th feature replaced by the background
            modified_sample = sample;
            modified_sample(j) = background(j);

            % Predict with the original and modified sample
            pred_original = predict(model, sample);
            pred_modified = predict(model, modified_sample);

            % Compute the contribution of the j-th feature
            shap_values(i, j) = pred_original - pred_modified;
        end
    end

    % Average over all samples to get global feature importance
    shap_values = mean(abs(shap_values), 1);
end