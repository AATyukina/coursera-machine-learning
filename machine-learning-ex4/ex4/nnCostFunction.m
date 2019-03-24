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

% Theta1 has size 25 x 401 
% Theta2 has size 10 x 26

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%feedforwart propagation
X = [ones(m, 1) X]; %X = 5000x401

sigm1 = sigmoid(X * Theta1'); %sigm1 = 5000x25
sigm1 = [ones(m, 1) sigm1]; %sigm1 = 5000x26


sigm2 = sigmoid(sigm1 * Theta2'); %5000x10
sigm2 = sigm2';

% теперь надо развернуть y и получить столбцы из векторов 
y_vec = zeros(num_labels, m); % 10*5000
for i=1:m,
  y_vec(y(i),i)=1; %на нужной строке нужного столбца ставим 1
end



J = (1/m) * sum ( sum ( (-y_vec) .* log(sigm2) - (1-y_vec) .* log(1-sigm2) )); %без регул€ризации
% 10x5000 .* 10x5000  -  10x5000.*10x5000
%теперь с регул€ризацией
% так как не учитываетс€ bias, то пропускаем первый столбец

J += (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
for t = 1:m, %дл€ каждого элемента отдельно
  %1 - feedforward
  a1 = X(t,:); % 1x401
  a1 = a1'; % 401x1
	z2 = Theta1 * a1; % 25x401*401x1
	a2 = sigmoid(z2); % 25x1
  
  a2 = [1 ; a2]; % 26x1
	z3 = Theta2 * a2; % 10x26*26x1
	a3 = sigmoid(z3);% 10 ч 1
  
  %2
  d_3 = a3 - y_vec(:, t);
  
  %3
  
  z2=[1; z2]; % 26x1
  d_2 = (Theta2' * d_3) .* sigmoidGradient(z2); %26x10 * 10x1 .* 26x1
  
  %4
  d_2 = d_2(2:end); % пропускаем нулевую сигма, так как bias  25x1

	Theta2_grad += d_3 * a2'; % 10x1*1x26
	Theta1_grad += d_2 * a1'; % 25x1*1x401

endfor

%находим как бы средний градиент


Theta2_grad = (1/m) * Theta2_grad; 
Theta1_grad = (1/m) * Theta1_grad; 





% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:, 2:end) += ((lambda/m) * Theta1(:, 2:end)); 
Theta2_grad(:, 2:end) += ((lambda/m) * Theta2(:, 2:end));
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
