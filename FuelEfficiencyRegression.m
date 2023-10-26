% Linear Regression for Fuel Efficiency Prediction
% Input data - Car attributes (X) and fuel efficiency values (y)
X = [1.8, 140, 1250, 8.2;
     2.0, 160, 1300, 7.5;
     1.6, 120, 1100, 9.0;
     2.2, 180, 1350, 7.2;
     1.5, 100, 1000, 10.0;
     2.5, 200, 1450, 6.5;
     1.8, 130, 1200, 8.5;
     2.0, 150, 1280, 7.8;
     1.6, 110, 1150, 9.2;
     2.3, 170, 1380, 7.0];

y = [22.5;
     24.8;
     20.3;
     25.1;
     19.5;
     26.2;
     21.8;
     24.1;
     20.7;
     25.9];

% Feature scaling
X_scaled = (X - mean(X)) ./ std(X);

% Add a column of ones as the intercept term
X_scaled = [ones(size(X_scaled, 1), 1), X_scaled];

% Initialize regression coefficients (theta)
theta = zeros(size(X_scaled, 2), 1);

% Hyperparameters
learning_rate = 0.01;
num_iterations = 1000;
epsilon = 0.0001;

% Gradient Descent
for iter = 1:num_iterations
    % Calculate hypothesis/predictions
    h = X_scaled * theta;

    % Calculate error
    error = h - y;

    % Calculate gradient
    gradient = (1 / size(X_scaled, 1)) * X_scaled' * error;

    % Update theta using gradient descent
    theta = theta - learning_rate * gradient;

    % Check convergence
    if max(abs(gradient)) < epsilon
        break;
    end
end

fprintf('Final theta values:\n');
disp(theta);

new_car = [1.9, 150, 1300, 8.0]; % Predict fuel efficiency for a new car
new_car_scaled = (new_car - mean(X)) ./ std(X);
new_car_scaled = [1, new_car_scaled];

predicted_fuel_efficiency = new_car_scaled * theta; % Prediction

fprintf('Predicted fuel efficiency for the new car: %.2f\n', predicted_fuel_efficiency);

