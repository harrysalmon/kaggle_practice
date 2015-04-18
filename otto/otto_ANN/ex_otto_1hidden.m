
% Initialization
clear ; close all; clc
cd('C:\Data\Data Science\Kaggle\otto\NN')
load logFile_1hidden.mat    % load log file
load('train.mat');          % load data (data is generated using the ex_prepareData script)


for lambda = [20]

% Setup the parameters you will use for this exercise
input_layer_size  = 93;  
hidden_layer_size = 400;
num_labels = 9;              
%lambda = 15;
rnd = [4545798, 5676787]; % random number generators for theta initialization
num_itrs = 2000;
m = size(X, 1);


% Initialize Theta
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size, rnd(1));
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels, rnd(2));


% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);


% Now, costFunction is a function that takes in only one argument (the neural network parameters)
options = optimset('MaxIter', num_itrs);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Predict
pred = predict(Theta1, Theta2, X);              % training set
pred_val = predict(Theta1, Theta2, X_val);      % validation set
pred_sub = predict(Theta1, Theta2, X_sub);      % submission set


% Calculate errors
err = -sum(sum((y_mat.*log(max(min(pred,1-1e-15),1e-15)))))/size(pred,1);
err_val = -sum(sum((y_mat_val.*log(max(min(pred_val,1-1e-15),1e-15)))))/size(pred_val,1);


% update log file & display result
logFile = [logFile; hidden_layer_size, lambda, num_itrs, rnd, err, err_val];
save('logFile_1hidden.mat', 'logFile')

end

format short g
logFile(:, [1:3,6:7])

% Prepare submission file (leave commented -- run in command line once you need to export result to csv)
%{
csvwrite('otto_submission_007_yyyymmdd.csv',[(1:size(pred_sub,1))' ,pred_sub])
header{1} = 'id'
for i=1:9
    header{i+1} = strcat('Class_',num2str(i));
end
csvwriteh('otto_submission_006_20150326.csv', [(1:size(pred_sub,1))' ,pred_sub], header)
%}
