function [ variance ] = variance( X )
% VARIANCE Calculates variance of vector.
%   Variance represents the sum of squared differences for each value from 
%   the mean value.

variance = sum((X - mean(X)) .^ 2);

end

