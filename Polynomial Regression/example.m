X = [1 2 3 4 5];
Y = [2 4 5 4 5];

[p, e] = polynomial_regression(X, Y, X, Y, 3);

plot(X, Y, 'o')
hold on
plot(p)
hold off
e