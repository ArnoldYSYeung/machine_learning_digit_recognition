function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%   Calculates the cost J and gradient grad of a neural network using the
%   regularized cost function.\
%   Input: nn_params - "unrolled" vectors representing weight parameters of
%                       each layer, theta1 and theta2
%          input_layer_size - # of nodes in the input layer (aka # of
%                               features) (e.g. 400)
%          hidden_layer_size -  # of nodes in the hidden layer (e.g. 25)
%          num_labels - # of nodes in the output layer (aka # of classes)
%                        (e.g. 10)
%          X - training dataset (rows = training examples, columns =
%                       features)
%          y - labels of training dataset (rows = labels, 1)
%          lambda - term used for regularization
%   Output: J - cost of neural network calculated using the regularized
%                  cost function
%           grad - optimized gradients of cost function (i.e. "unrolled"
%                  Theta1 and Theta2)
%
%   Written by Arnold Yeung
%   Date: June 29, 2013
%   arnoldyeung.com


% Reshape nn_params back into the Theta1 and Theta2, the weight matrices
% for the 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables

m = size(X, 1);                 % number of rows (training examples)
         
J = 0;                                  % cost
Theta1_grad = zeros(size(Theta1));      % gradient of theta 1
Theta2_grad = zeros(size(Theta2));      % gradient of theta 2

%% PART 1: FORWARD PASS
%   Calculate the cost J using inputted Theta1 and Theta2

% Input Layer
a1 = [ones(m, 1) X];                     % add column of ones to rows(training examples)

% Hidden Layer (activations for layer 2)
z2 = Theta1 * a1';
[m_z2, n_z2] = size(z2);                 % define n_z2 as the # of columns (training examples)
a2 = [ones(1, n_z2); sigmoid(z2)];       % add row of ones (training examples)

% Output Layer (activations for layer 3)
z3 = Theta2 * a2;
a3 = sigmoid(z3);
h = a3;                                  % probability of a training set matching each class

% convert y to y_k 

y_k = zeros(m, num_labels);              % 5000 x 10 matrix representing y using 0 and 1's (k is column)

for i = 1:m                              % for each training example
    y_k(i, y(i)) = 1;                    % fill in k column with 1
end

y_k = y_k';                 % convert each COLUMN to represent one training example
                            % this is a 10 * 5000 matrix where (a, b) = 1...
                            % where training example b is classified in
                            % class a
                           
% Accumulate cost
sum = 0;

for i = 1:m                 % for each training example
    for k = 1:num_labels    % for each class
        sum = -y_k(k,i)*log(h(k,i)) - (1-y_k(k,i))*log(1-h(k,i));
        J = J + sum; % this gives the acculmated cost, a scalar value
    end
end
        
J = J/m;                    % mean cost 

% regularization of cost 

reg = 0;        % regularization term
sum_theta1 = 0;
sum_theta2 = 0;

for j = 1:hidden_layer_size
    for k = 2:(input_layer_size+1)  % add 1 because there is extra column ("the bias")
        sum_theta1 = sum_theta1 + Theta1(j,k)^2;
    end
end

for j = 1:num_labels
    for k = 2:(hidden_layer_size+1) % add 1 because there is extra column ("the bias")
        sum_theta2 = sum_theta2 + Theta2(j,k)^2;
    end
end

reg = sum_theta1 + sum_theta2;      % calculate the regularization term

J = J + reg*lambda/(2*m);           % regularized cost

%% PART 2: BACK PROPAGATION
%   Solve for the gradient

% define size of Delta1, Delta2 (accumulated gradient)
Delta1 = zeros(size(Theta1_grad, 1), size(Theta1_grad, 2));
Delta2 = zeros(size(Theta2_grad, 1), size(Theta2_grad, 2));



for t = 1:m             % for each training example
    % forward pass
    a1 = [1 , X(t,:)];  % 1 x 401; convert mth row of X to a1 with bias unit (1) at front
    z2 = Theta1*a1';    % 25 x 1
    
    a2 = [1; sigmoid(z2)];  % 26 x 1
    z3 = Theta2 * a2;   % 10 x 1
    
    a3 = sigmoid(z3);   % 10 x 1
    h = a3;
    
    % convert y to y_k
    y_k = ([1:num_labels]==y(t))';
    % if y_k = 1, belongs in class k
    % if y_k = 0, does not belongs in class k
    % for example if y(t) = 5, then y_k = [0 0 0 0 1 0 0 0 0 0]'
    
    % compute delta3, delta2, Delta (accumulator)
    delta3 = a3 - y_k;  % 10 x 1
    
    delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];    % 26 x 1
    delta2 = delta2(2:end);      % remove bias unit
    
    Delta2 = Delta2 + delta3*a2';       
    Delta1 = Delta1 + delta2*a1;
end

% unregularized (mean) gradients
Theta1_grad = Delta1/m;     
Theta2_grad = Delta2/m;

% Regularization (Change the terms where j>= 1)

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * Theta2(:,2:end));

% Concatenate gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

