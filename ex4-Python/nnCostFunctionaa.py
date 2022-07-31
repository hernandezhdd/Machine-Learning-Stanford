def nnCostFunction(nn_params, input_lyr_sz, hidd_lyr_sz, num_lbls, X, y, lmbd):
                                   
    # %NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # %neural network which performs classification
    # %   [J grad] = NNCOSTFUNCTON(nn_params, hidd_lyr_sz, num_lbls, ...
    # %   X, y, lmbd) computes the cost and gradient of the neural network. The
    # %   parameters for the neural network are "unrolled" into the vector
    # %   nn_params and need to be converted back into the weight matrices. 
    # % 
    # %   The returned parameter grad should be a "unrolled" vector of the
    # %   partial derivatives of the neural network.
    # %
    # % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # % for our 2 layer neural network
    
    import numpy as np
    
    Theta1 = np.reshape(nn_params[0:hidd_lyr_sz * (input_lyr_sz + 1)], (hidd_lyr_sz, input_lyr_sz + 1))

    Theta2 = np.reshape(nn_params[hidd_lyr_sz * (input_lyr_sz + 1):], (num_lbls, hidd_lyr_sz + 1))

    # % Setup some useful variables
    m = X.shape[0]

    # % You need to return the following variables correctly 

    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # % Add ones to the X data matrix
    
    X = np.append(np.ones((m,1)), X, axis=1)
    
    # tic #QUE HAGO CON ESTO

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: You should complete the code by working through the
    # %               following parts.
    # %
    # % Part 1: Feedforward the neural network and return the cost in the
    # %         variable J. After implementing Part 1, you can verify that your
    # %         cost function computation is correct by verifying the cost
    # %         computed in ex4.m
    # %

    # labelsVec = np.append(np.zeros((k,1)), 1)
    # labelsVec = np.append(labelsVec, np.zeros((num_lbls-k-1, 1)))
    # labelsVec = labelsVec.T
        
    import numpy.matlib

    # yVec = np.zeros (( len(y), 1, num_lbls+1))

    yVec = np.matlib.repmat(y, 1,num_lbls+1)

    for k in range(1,num_lbls+1):
        yVec[:,k] = np.where( yVec[:,k]==k, 1,0 )
        
        # idxs = np.find( y==k )
        
        # WTFFF
        # yVec[ idxs, : ] = repmat(labelsVec, length(idxs), 1)

    # Variables' sizes X 5000X401 yVec 5000x10, Theta1 25x401, Theta2 10X26
    # a2 5000X25 a3 

    z2 = np.dot(X, Theta1.T)
    
    from sigmoid import sigmoid

    a2 = sigmoid( z2)
    # [np.ones(m, 1) X]
    a2 = np.append(np.ones((a2.shape[0],1)), a2, axis=1)
    # a2 = [np.ones(a2.shape[0], 1) a2]; # a2 5000X26 

    #h_theta
    z3 = np.dot(a2, Theta2.T)

    a3 = sigmoid( z3); # a3 5000X10
    
    J = (1- yVec) * np.log (1 - a3)
    
    J = J + yVec * np.log (a3)

    J = - 1/m * np.sum (np.sum( J ))

    J = J + 0.5*lmbd/m*  np.sum( np.sum( Theta1[:,1:]**2)) 
                         
    J = J + np.sum( np.sum( Theta2[:,1:]**2))

    # % Part 2: Implement the backpropagation algorithm to compute the gradients
    # %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    # %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    # %         Theta2_grad, respectively. After implementing Part 2, you can check
    # %         that your implementation is correct by running checkNNGradients
    # %
    # %         Note: The vector y passed into the function is a vector of labels
    # %               containing values from 1..K. You need to map this vector into a 
    # %               binary vector of 1's and 0's to be used with the neural network
    # %               cost function.
    # %
    # %         Hint: We recommend implementing backpropagation using a for-loop
    # %               over the training examples if you are implementing it for the 
    # %               first time.
    # %

    # Variables' sizes X 5000X401 yVec 5000x10, Theta1 25x401, Theta2 10X26
    # a2 5000X25 a3 5000X10

#     Delta1 = zeros(hidd_lyr_sz, input_lyr_sz+1);

#     Delta2 = zeros(num_lbls, hidd_lyr_sz+1);

#     z2 = X * Theta1.T;

#     a2 = sigmoid( z2);

#     a2 = [ones(size(a2, 1), 1) a2]; # a2 5000X26 

#     z3 = a2 * Theta2';

#     a3 = sigmoid(z3); # a3 5000X10

#     delta3 = a3 - yVec;

#     ##           26X10      10X5000          5000X26
#     delta2 = (  (Theta2' * delta3')' .* ( (a2.*( 1 - a2 )) )) ;

#     Delta2 = Delta2 + delta3' * a2;

#     delta1 = sum( transpose(Theta1(:,2:end)' * delta2(:,2:end)') .* ( (X(:,2:end).*( 1 - X(:,2:end) )) ));

#     Delta1 = Delta1 + delta2(:,2:end)' * X;  


#     Theta1_grad = Delta1 / m;

#     Theta2_grad = Delta2 / m;

#     % Part 3: Implement regularization with the cost function and gradients.
#     %
#     %         Hint: You can implement this around the code for
#     %               backpropagation. That is, you can compute the gradients for
#     %               the regularization separately and then add them to Theta1_grad
#     %               and Theta2_grad from Part 2.
#     %
#     Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lmbd/m * Theta1(:,2:end);

#     Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lmbd/m * Theta2(:,2:end);

#     % -------------------------------------------------------------

#     % =========================================================================

#     % Unroll gradients
#     grad = [Theta1_grad(:) ; Theta2_grad(:)];

#     tiempos=toc;

    # return [J, grad, tiempos]

    return J
                