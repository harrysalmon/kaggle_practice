function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));

% output function dimensions m, num_labels

a_1 = [ones(m,1), X]; % m x 401
z_2 = a_1*Theta1';   % m x 401 x 401 x 25 = m x 25
a_2 = [ones(m,1), sigmoid(z_2)]; % m x 26
z_3 = a_2* Theta2'; % m x 26 x 26 x 10 % m X 10 
a_3 = sigmoid(z_3); % m X 10
% yy = repmat(y,1, num_labels)==repmat((1:num_labels),m,1);
yy = zeros(m,num_labels);
yy(sub2ind(size(yy),1:m,y')) = 1;
J = sum(sum(-yy.*log(a_3) - (1-yy).*log((ones(m, num_labels)- a_3))))/m + lambda*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))/(2*m);

% grad=0;

delta3 = a_3-yy; % m x 10
delta2 = (delta3*Theta2).*[ones(m,1), sigmoidGradient(z_2)]; % m x 26

Theta2_grad = delta3'* a_2/m; % 10 x m x m x 26
Theta1_grad = delta2(:,2:end)'*a_1/m;
 
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda*Theta2(:,2:end)/m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda*Theta1(:,2:end)/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

