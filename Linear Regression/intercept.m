function [ b0 ] = intercept( X, Y, slope )
% INTERCEPT Calculates intercept of regression line.
%   Intercept represents the point on y axis that always has value 0 for x
%   axis. That is starting point of our regression line.

b0 = mean(Y) - slope .* mean(X);

end

