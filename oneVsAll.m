function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%   Trains logistic regression for multiple classifiers and returns all
%   weight parameters of classifiers in a matrix all_theta
%   Input: X - training dataset (row = training examples, columns =
%                   features)
%          y - training labels (row = training examples, 1)
%          num_labels - number of classes (e.g. 10)
%          lambda - term for regularization
%   Output: all_theta - weight parameters of logistic regression
%               classifiers where each row corresponds to a classifier
%               (e.g. row 2 corresponds with classifier for class 2)
%
%   Written by Arnold Yeung
%   Date: June 1, 2013
%   arnoldyeung.com


% Some useful variables
m = size(X, 1);         % number of rows (i.e. number of training examples)
n = size(X, 2);         % number of columns (i.e. number of features/pixels) 

% Add ones to the X data matrix
X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);            % initial theta, num of features x 1
all_theta = zeros(num_labels, n+1);         % num of classes (10) x (num of features + 1)

% NOTE: y is the test data.  It is not the TRUE data.

for k = 1:num_labels        % for each class
  
  y_true(:,k) = (y == k);   % test whether y = k (returns y_true(k) = 1 if yes, y_true(k) = 0 if no)
  
  % columns (n) represent the num_label (class) that the training example fits with.  
  % rows (m) represent the training example.
  % e.g. [0 0 0 1 0 0 0 0 0 0] means that this training example fits with num_label = 4       
  
  
  initial_theta = zeros(n + 1, 1);      % starting classifier parameter for class (num of features x 1)
  % Setting 'options' for fmincg
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  
  % @ sets theta = lrCostFunction(theta, X, y_true(:,k), lambda) <-recursion
  % start with initial_theta = theta
  all_theta(k,:) = fmincg(@(theta)(lrCostFunction(theta, X, y_true(:,k), lambda)), initial_theta, options);
end

