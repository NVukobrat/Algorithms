function [ X_vender ] = vendermonde_matrix( X, degree )
%VENDERMONDE_MATRIX Implementation of vendermonde matrix.
%   Vendermonde matrix is a matrix with the terms of a geometrics
%   progression in each row of MxN matrix.

X_vender = zeros(length(X), degree + 1);
X_vender(:, 1) = ones(length(X), 1); % 0 degree

for i = 1:degree
    X_deg = (X .^ i) .';
    X_vender(:, i + 1) = X_deg;
end
