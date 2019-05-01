function [ predictions, error_score ] = linear_regression( X_train, Y_train, X_test, Y_test )
% LINEAR_REGRESSION Trains and returns prediction for test data.

[b0, b1] = coefficients(X_train, Y_train);
predictions = predict(b0, b1, X_test);
error_score = rmse_metric(predictions, Y_test);

end

