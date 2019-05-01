function [ covariance ] = covariance( X, Y )
% COVARIANCE calculates covariance of two groups of values.
%   Covariance describes how two gorups of numbers changes together. It
%   represents the corelation between them.

covariance = sum((X - mean(X)) * (Y - mean(Y)).');

end

