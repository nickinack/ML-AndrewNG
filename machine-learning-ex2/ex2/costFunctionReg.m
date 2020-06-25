function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


x_pred =  X * theta;
y_pred = sigmoid(x_pred);
cost = (-y' * log(y_pred)) - ((1-y)' * log(1 - y_pred));
J = cost/m + ((lambda/(2*m)) * sum(theta(2:end).^2)) ; 

err = sigmoid(x_pred) - y ;
theta_model = theta;
theta_model(1) = 0
grad = ((1/m) * (X' * err)) + ((lambda/m)*theta_model);





% =============================================================

end
