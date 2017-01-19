%   This script runs multi-class logistic regression and neural network
%   with regularization for a dataset containing the pixels of handwritten
%   digits.  The dataset used is a subset of MNIST handwritten digits
%   (http://yann.lecun.com/exdb/mnist).
%
%   This project is based on Exercises 3 and 4 in Coursera course, Machine
%   Learning by Andrew Ng, Stanford University
%   (https://www.coursera.org/learn/machine-learning).

%   All scripts and functions attached in this project, with the following
%   exceptions, were written by Arnold Yeung:
%       - fmincg.m
%       - displayData.m
%
%   Written by Arnold Yeung
%   Date: January 19, 2017
%   arnoldyeung.com

%%  Initialization

clear; clc; close all;

num_labels = 10;                % number of classes

%%  Create training and test sets
% Load Training Data
fprintf('Loading and Visualizing Data ...\n');
load('handwritten.mat');

m = length(y);                  % number of examples
shuffle = randperm(m);          % shuffle order of examples
shuffledX = X(shuffle, :);      % shuffle features
shuffledy = y(shuffle, :);      % shuffle labels ACCORDINGLY

% take top 4000 examples as training set
trainX = shuffledX(1:4000,:);
trainy = shuffledy(1:4000,:);

% take last 1000 examples as test set
testX = shuffledX(4001:5000,:);
testy = shuffledy(4001:5000,:);

% Randomly select 100 data points to display
sel = shuffledX(1:100, :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Logistic Regression
fprintf('\nTraining One-vs-All Logistic Regression...\n')

% train
lambda = 0.1;           % original 0.1
[all_theta] = oneVsAll(trainX, trainy, num_labels, lambda);   % all_theta is trained classifier

fprintf('Program paused. Press enter to continue.\n');
pause;

% training set prediction
trainLrPred = predictOneVsAll(all_theta, trainX);       % prediction of classes

trainLrAcc = mean(double(trainLrPred == trainy)) * 100;     % accuracy
fprintf('\nLogistic Regression Training Set Accuracy: %f\n', trainLrAcc);

% test set prediction
testLrPred = predictOneVsAll(all_theta, testX);       % prediction of classes

testLrAcc = mean(double(testLrPred == testy)) * 100;   % accuracy
fprintf('\nLogistic Regression Test Set Accuracy: %f\n', testLrAcc);

fprintf('Program paused. Press enter to continue.\n');
pause;

%%  Neural Network

% Layer sizes
[numTrain, numFeats] = size(trainX);
input_layer_size  = numFeats;               % 20x20 Input Images of Digits
hidden_layer_size = 25;                     % 25 hidden units

fprintf('\nInitializing Neural Network Parameters ...\n')

% create initial Theta1 and Theta2 to start with for optimization
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);          % try different # of iterations
lambda = 1;                                 % try different values

% "Short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, trainX, trainy, lambda);

% minimize cost and determine optimal Theta1 and Theta2 <- recursion
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Convert ("reroll") Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
fprintf('Program paused. Press enter to continue.\n');
pause;

% Visualize weight parameters Theta1 and Theta2
fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));
fprintf('\n Display of Theta1.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
displayData(Theta2(:, 2:end));
fprintf('\n Display of Theta2.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% training set prediction
trainNnPred = nnPredict(Theta1, Theta2, trainX);    % predicted classes
trainNnAcc = mean(double(trainNnPred == trainy)) * 100;    % accuracy

fprintf('\nNeural Network Training Set Accuracy: %f\n', trainNnAcc);

% training set prediction
testNnPred = nnPredict(Theta1, Theta2, testX);    % predicted classes
testNnAcc = mean(double(testNnPred == testy)) * 100;    % accuracy

fprintf('\nNeural Network Test Set Accuracy: %f\n', testNnAcc);

fprintf('Program paused. Press enter to continue.\n');
pause;

%%  Summary of Accuracy Results

fprintf('\nLogistic Regression Training Set Accuracy: %f\n', trainLrAcc);
fprintf('\nLogistic Regression Test Set Accuracy: %f\n', testLrAcc);
fprintf('\nNeural Network Training Set Accuracy: %f\n', trainNnAcc);
fprintf('\nNeural Network Test Set Accuracy: %f\n', testNnAcc);


