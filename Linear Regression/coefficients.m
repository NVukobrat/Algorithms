function [ b0, b1 ] = coefficients( X, Y )
% COEFICIENTS Calculates needed coeficients.
%   Coeficients for linear regression are slope and intercept. They are
%   used to determent the regression line and estimate the predictions.

b1 = slope(X, Y);
b0 = intercept(X, Y, b1);

end
