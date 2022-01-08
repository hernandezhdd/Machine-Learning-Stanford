function [C, sigma, errorsVec] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% You should try to change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)

models = containers.Map(); 
preds = containers.Map(); 
val_errors = containers.Map(); 

cVec = [0.01, 0.1, 1, 10, 100];

sigmaVec = [0.01, 0.1, 1, 10, 100];

%cVec = [0.01, 0.1];

%sigmaVec = [0.01];

errorsVec = [];

for sigma_i = sigmaVec

    for C_i = cVec
        
        printf('Training for C and sigma '), disp([C_i, sigma_i])
        
        model = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i));
        
        models(strcat(num2str(C_i),',', num2str(sigma_i))) = model;
        
        predictions = svmPredict(model, Xval);
        
        preds(strcat(num2str(C_i),',', num2str(sigma_i))) = predictions;
        
        val_errors(strcat(num2str(C_i),',', num2str(sigma_i))) = mean(double(predictions ~= yval));
        
        errorsVec = [errorsVec; [C_i, sigma_i, mean(double(predictions ~= yval))] ];
    end
end

% find min value of val_errors 
% from mathworks
%https://www.mathworks.com/matlabcentral/answers/98444-how-can-i-retrieve-the-key-which-belongs-to-a-specified-value-using-the-array-containers-map-in-matl

testvalue = min([values(val_errors){:} ]);

testind = cellfun(@(x)isequal(x,testvalue), values(val_errors) );

testkeys = keys(val_errors);

msg_key = testkeys(testind){1};

C = str2num( strsplit (msg_key, ","){1});
sigma = str2num(strsplit (msg_key, ","){2});

% =========================================================================

end
