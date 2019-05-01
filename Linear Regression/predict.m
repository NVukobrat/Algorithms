function [ predictions ] = predict( b0, b1, X_test )
% PREDICT Estemates values for X.

predictions = b0 + b1 .* X_test;

end

