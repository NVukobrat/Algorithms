%% Initialize
clear; close all; clc

%% Load data
data = load('Data.txt');
X = data(:, 1:2);
y = data(:, 3);

%% Compute cost and gradient
[m, n] = size(X);
initial_theta = zeros(n + 1, 1);

[cost, grad] = computeCost(initial_theta, X, y);

%% Optimize function
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(computeCost(t, X, y)), initial_theta, options);

%% Make prediction
predictions = predict(theta, X);
accuracy = mean((predictions == y) .* 100);