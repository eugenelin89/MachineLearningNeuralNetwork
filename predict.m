function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m, 1) X];
Z = Theta1 * X'; % z dimension 25x5000.  Each row is an activation unit, across traing examples.  Each column is a training example.
A = sigmoid(Z); % same dimension of Z.
% At this point, we've determined the values for activation unit of hidden layer, a_1_1, a_2_2,...a_25_2 for the 5000 training examples.  Each training exampleis a column.
% We invert A, then this is just like X again.
A = A';
A = [ones(m, 1) A];

Z2 = Theta2 * A'; 
H = sigmoid(Z2);% H dimension is 10x5000.  Each row is for each class, and each column is for each training example.

[val, p] = max(H);

% we can optimize by not doing A = A' above and add a row vector of 1 in the previous step.  But for clarity, we stick with this for now.









% =========================================================================


end
