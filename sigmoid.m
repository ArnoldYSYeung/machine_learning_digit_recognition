function g = sigmoid(z)
%   Compute the sigmoid function.
%   Input: z - vector or scalar value
%   Output: g - vector or scalar value
%   
%   Written by Arnold Yeung
%   Date: June 29, 2013
%   arnoldyeung.com

g = 1.0 ./ (1.0 + exp(-z));

