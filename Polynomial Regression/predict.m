function [ predictions ] = predict( coefficients, X, degree )
% PREDICT Estemates values for X.

X = vendermonde_matrix(X, degree);
predictions = (X * coefficients.').';

end

