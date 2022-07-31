function [J grad tiempos] = nnCostFunction(nn_params, ...
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

##[coso, temp] = max(a3, [], 2);
##p = temp; 

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Add ones to the X data matrix
X = [ones(m, 1) X];

tic

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

yVec = zeros ( size(y), num_labels) ;

for k=1:num_labels
  
labelsVec = transpose([ zeros(k-1,1) ; 1 ; zeros(num_labels-k, 1)]);

idxs = find( y==k );

yVec( idxs, : ) = repmat(labelsVec, length(idxs), 1);

end


# Variables' sizes X 5000X401 yVec 5000x10, Theta1 25x401, Theta2 10X26
# a2 5000X25 a3 

z2 = X * Theta1';

a2 = sigmoid( z2);

a2 = [ones(size(a2, 1), 1) a2]; # a2 5000X26 

#h_theta
z3 = a2 * Theta2';

a3 = sigmoid( z3); # a3 5000X10

J = - 1/m * sum (sum(yVec .* log (a3) + (1- yVec) .* log (1 - a3)));

J = J + 0.5*lambda/m* ( sum(sum( Theta1(:,2:end).**2)) + sum(sum( Theta2(:,2:end).**2))) ;



##grad(1) = 1/m * (( h_theta(:,1)' - y(:,1)') * X(:,1));

##grad = 1/m * (( h_theta' - y') * X + lambda/m*theta)';

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

# Variables' sizes X 5000X401 yVec 5000x10, Theta1 25x401, Theta2 10X26
# a2 5000X25 a3 5000X10

##Delta1 = zeros(hidden_layer_size, input_layer_size);
##
##Delta2 = zeros(num_labels, hidden_layer_size);

Delta1 = zeros(hidden_layer_size, input_layer_size+1);

Delta2 = zeros(num_labels, hidden_layer_size+1);


  tic
  z2 = X * Theta1';

  a2 = sigmoid( z2);

  a2 = [ones(size(a2, 1), 1) a2]; # a2 5000X26 

  z3 = a2 * Theta2';

  a3 = sigmoid(z3); # a3 5000X10

  delta3 = a3 - yVec;
 
##           26X10      10X5000          5000X26
  delta2 = (  (Theta2' * delta3')' .* ( (a2.*( 1 - a2 )) )) ;

  Delta2 = Delta2 + delta3' * a2;

  delta1 = sum( transpose(Theta1(:,2:end)' * delta2(:,2:end)') .* ( (X(:,2:end).*( 1 - X(:,2:end) )) ));

  Delta1 = Delta1 + delta2(:,2:end)' * X;  

Theta1_grad = Delta1 / m;
  
Theta2_grad = Delta2 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

tiempos=toc;

end
