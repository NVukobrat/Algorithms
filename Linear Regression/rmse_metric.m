function [ error_score ] = rmse_metric( predictions, actual )
% RMSE_METRIC Root mean squared error.
%   Mesures average magnitude of the error. It represents the squared root
%   ot the average of squared differences between predictions and actual
%   observations.

error_score = sqrt(sum((predictions - actual) .^ 2) / length(predictions));

end

