function [ b1 ] = slope( X, Y )
% SLOPE Calculates slope of regression line.
%   Slope represents the steepness of the regression line. It describes the
%   direction of line, does it grows or lowers.

b1 = covariance(X, Y) / variance(X);

end
