function [ cost, grad ] = computeCost( theta, X, y )

m = size(X, 1);
X = [ones(m, 1) X];
grad = zeros(size(theta));

h = sigmoid(X * theta);
cost = -(1 / m) * sum( (y .* log(h)) + ((1 - y) .* log(1 - h)) );

for i = 1:size(theta, 1)
    grad(i) = (1 / m) * sum( (h - y) .* X(:, i) );
end

end

