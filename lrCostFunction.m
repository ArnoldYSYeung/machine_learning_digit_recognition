function [J, grad] = lrCostFunction(theta, X, y, lambda)

%   Compute the cost and gradient for logistic regression using the cost
%   function with regularization using theta as the weight parameter.
%   Input: theta - weight parameter of logistic regression
%          X - training dataset (row = training examples, columns =
%                   features)
%          y - training labels (row = training examples, 1)
%          lambda - term for regularization
%
%   Written by Arnold Yeung
%   Date: June 1, 2013
%   arnoldyeung.com

format long;

% Initialize some useful values
m = length(y); % number of training examples

J = 0;                          % cost set to 0
grad = zeros(size(theta));      % gradient set to 0

% determine the cost of the regularized cost function
z = X*theta;
h = sigmoid(z);
J = sum(-y.*log(h) - (1-y).*log(1-h))/m + lambda*sum(sum(theta(2:end).^2))/(2*m);

% determine the regularized gradient
grad = X'*(h-y)./m;
grad(2:end) = grad(2:end) + lambda*theta(2:end)/m;

