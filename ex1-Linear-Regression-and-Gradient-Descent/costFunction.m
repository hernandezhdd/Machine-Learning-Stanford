function [jVal, gradient] = costFunction(theta, X, y)
##  jVal = #[...code to compute J(theta)...];
  
  m = length(y); % number of training examples
  preds = theta' * X';

  jVal = 0.5/m * sum((preds' - y ).**2);   
  
##  gradient = [...code to compute derivative of J(theta)...];

  gradient = ( 1/m * sum (( preds' - y ) .* X ) )' ;

end