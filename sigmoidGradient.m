function g = sigmoidGradient(z)
%   Computes the gradient of the sigmoid function for input z
%   Input:  z - input vector or scalar
%   Output: g - output vector or scalar
%
%   Written by Arnold Yeung
%   Date: June 29, 2013
%   arnoldyeung.com

g = zeros(size(z));
g = sigmoid(z).*(1-sigmoid(z));

