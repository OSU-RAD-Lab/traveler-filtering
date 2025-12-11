clear; clc; close all;

% Link lengths (m)
L1 = 0.09;  % Upper leg
L2 = 0.09;  % Lower leg

% Motor step resolution (degrees → radians)
step_deg = 3;
step_rad = deg2rad(step_deg);

% Target path (in leg plane)
n_points = 100;
x_path = linspace(-0.08, 0.08, n_points);  % horizontal motion
y_path = -0.12 * ones(1, n_points);        % fixed height

% Initialize outputs
theta1_out = zeros(1, n_points);
theta2_out = zeros(1, n_points);
sol_found = false(1, n_points);

% IK Solver with Multiple Guesses (elbow-up and elbow-down)
for i = 1:n_points
    x = x_path(i);
    y = y_path(i);
    
    % Distance from hip to foot
    r = sqrt(x^2 + y^2);
    
    % Check reachability
    if r > (L1 + L2) || r < abs(L1 - L2)
        warning("Target point unreachable at step %d", i);
        theta1_out(i) = NaN;
        theta2_out(i) = NaN;
        continue;
    end
    
    % Compute inner angles via Law of Cosines
    cos_theta2 = (x^2 + y^2 - L1^2 - L2^2) / (2 * L1 * L2);
    theta2a = acos(cos_theta2);         % Elbow-down
    theta2b = -acos(cos_theta2);        % Elbow-up
    
    % Solve for theta1
    k1 = L1 + L2 * cos(theta2a);
    k2 = L2 * sin(theta2a);
    theta1a = atan2(y, x) - atan2(k2, k1);
    
    k1 = L1 + L2 * cos(theta2b);
    k2 = L2 * sin(theta2b);
    theta1b = atan2(y, x) - atan2(k2, k1);
    
    % Now apply motor resolution: snap each to closest step
    th1a_snapped = round(theta1a / step_rad) * step_rad;
    th2a_snapped = round(theta2a / step_rad) * step_rad;
    th1b_snapped = round(theta1b / step_rad) * step_rad;
    th2b_snapped = round(theta2b / step_rad) * step_rad;

    % Forward kinematics for both guesses
    xa = L1 * cos(th1a_snapped) + L2 * cos(th1a_snapped + th2a_snapped);
    ya = L1 * sin(th1a_snapped) + L2 * sin(th1a_snapped + th2a_snapped);
    err_a = norm([xa - x; ya - y]);

    xb = L1 * cos(th1b_snapped) + L2 * cos(th1b_snapped + th2b_snapped);
    yb = L1 * sin(th1b_snapped) + L2 * sin(th1b_snapped + th2b_snapped);
    err_b = norm([xb - x; yb - y]);

    % Choose better solution
    if err_a < err_b
        theta1_out(i) = th1a_snapped;
        theta2_out(i) = th2a_snapped;
    else
        theta1_out(i) = th1b_snapped;
        theta2_out(i) = th2b_snapped;
    end
    
    sol_found(i) = true;
end

% Plotting joint angles
figure;
subplot(2,1,1);
plot(rad2deg(theta1_out), 'b'); hold on;
plot(rad2deg(theta2_out), 'r');
xlabel('Time step'); ylabel('Angle (deg)');
legend('Hip \theta_1','Knee \theta_2');
title('Joint Angles with Step Resolution');

% Plot actual foot tip positions using FK
x_actual = L1*cos(theta1_out) + L2*cos(theta1_out + theta2_out);
y_actual = L1*sin(theta1_out) + L2*sin(theta1_out + theta2_out);

subplot(2,1,2);
plot(x_path, y_path, 'k--', 'DisplayName', 'Desired');
hold on;
plot(x_actual, y_actual, 'b-', 'DisplayName', 'Achieved');
xlabel('X (m)'); ylabel('Y (m)');
legend();
axis equal;
title('Foot Tip Position (Achieved vs Desired)');
