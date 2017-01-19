function p = predictOneVsAll(all_theta, X)
%   Predicts the classes of dataset X for trained multi-class logistic
%   regression classifiers based on weight parameters of all_theta.
%   Input: X - training dataset (row = training examples, columns =
%                   features)
%          all_theta - weight parameters of logistic regression
%               classifiers where each row corresponds to a classifier
%               (e.g. row 2 corresponds with classifier for class 2)
%   Output: p - predicted classes of dataset X (row = training example, 1)
%
%   Written by Arnold Yeung
%   Date: June 1, 2013
%   arnoldyeung.com


m = size(X, 1);                     % number of training examples
num_labels = size(all_theta, 1);    % number of classes

p = zeros(size(X, 1), 1);           % predictions of all training examples

% Add ones to the X data matrix
X = [ones(m, 1) X];                 

z = all_theta * X';
h = sigmoid(z);                 % probability of training example matching classes
% row(m) is class
% column(n) is training example

% 1st column of prob matrix indicates whether a column (training example) has
% its max probability in class 1
for c = 1:num_labels
    prob(:,c) = (h(c,:) == max(h));
end

% maximum indicates the maximum value in a row
% j indicates the column that maximum value is in
[maximum, p] = max(prob, [], 2);    % we need j