function p = nnPredict(Theta1, Theta2, X)
%   Predicts the label of inputted X given trained weights Theta1 and Theta2
%   of a neural network
%   Input: Theta1 - weight parameters of layer 1 and layer 2
%          Theta2 - weight parameters of layer 2 and layer 3
%          X - dataset to be classified (rows = training examples, columns
%                   = features)
%   Output: p - predicted classes of X (rows = training examples, 1)
%
%   Written by Arnold Yeung
%   Date: June 29, 2013
%   arnoldyeung.com

% Useful values
m = size(X, 1);                     % number of training examples
num_labels = size(Theta2, 1);       % number of classes

% predicted classification
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');        
h2 = sigmoid([ones(m, 1) h1] * Theta2');     % calculate probability of each class
[dummy, p] = max(h2, [], 2);                 % predicted class

