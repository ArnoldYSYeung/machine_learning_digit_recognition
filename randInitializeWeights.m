function weights = randInitializeWeights(L_in, L_out)
%   Produces a set of random weights for initial_Thetas.  Weights are
%   randomly determined based on L_in and L_out.
%   Input: L_in - number of incoming connections
%          L_out - number of outgoing connections
%   Output: weights - weights of a layer with L_in and L_out
%   
%   Written by Arnold Yeung
%   Date: June 29, 2013
%   arnoldyeung.com

% set size of weights
weights = zeros(L_out, 1 + L_in); 

% Initialize weights randomly to break symmetry
% Note: The first row of W corresponds to the parameters for the bias units

% RULE OF THUMB: epsilon_init = sqrt(6)/sqrt(L_in + L_out)
epsilon_init = 0.12;    
    
% randomly calculate weights for theta (classifier)
weights = rand(L_out, 1+L_in) * 2 * epsilon_init - epsilon_init; 