function [ predictions, error_score ] = polynomial_regression( X_train, Y_train, X_test, Y_test, degree )
% POLYNOMIAL_REGRESSION Implementation of polynomial regression.
%   Polynomial Regression describes relationship between the independent
%   variable x and dependent variable y modelled as an nth degree of
%   polynomial x.

[coefficients_array] = coefficients(X_train, Y_train, degree);
predictions = predict(coefficients_array, X_test, degree);
error_score = rmse_metric(predictions, Y_test);

end

