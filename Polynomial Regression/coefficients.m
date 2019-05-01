function [ coefficients_array ] = coefficients( X, Y, degree )
%COEFFICIENTS Calculates needed coeficients.
%   Coefficients for Polynomial Regression represent estimated polynomial
%   regression coefficients using ordinary least squares estimation. For
%   calculation it uses Vendermonde matrix.
%   This estimation assumes that degree < number of data points.

if (degree > length(Y))
    error('Degree needs to be bigger than number of data points')
end

X = vendermonde_matrix(X, degree);
coefficients_array = (((X.' * X) ^ -1) * X.' * Y.')';

end

