
clear; clc;

% Parameters
res_deg = 3;
dt = 0.01;        % Time step (s)
L = 10;          % Number of steps
time = (0:L-1) * dt;

% Motion path: horizontal line 10 cm wide at y = -0.15
x_path = linspace(-0.05, 0.05, L);
y_path = -0.21 * ones(1, L);

% Preallocate
theta1_q = zeros(1, L);
theta2_q = zeros(1, L);
ideal_thetas = zeros(2, L);
actual_tip = zeros(2, L);

% Inverse kinematics with motor resolution
for i = 1:L
    [th1, th2, tip, ~, ideal] = ik_with_motor_resolution(x_path(i), y_path(i), res_deg);
    theta1_q(i) = th1;
    theta2_q(i) = th2;
    actual_tip(:, i) = tip;
    ideal_thetas(:, i) = ideal;
end

% Estimate velocities and accelerations
theta1_dot = [0 diff(theta1_q)] / dt;
theta2_dot = [0 diff(theta2_q)] / dt;
theta1_ddot = [0 diff(theta1_dot)] / dt;
theta2_ddot = [0 diff(theta2_dot)] / dt;

% Motor inertial and damping properties
I = 0.002;       % Moment of inertia (kg*m^2)
b = 0.01;        % Damping coefficient (N*m*s/rad)

% Torque estimate
torque1 = I * theta1_ddot + b * theta1_dot;
torque2 = I * theta2_ddot + b * theta2_dot;

% Filtering torque (low-pass filter)
alpha = 0.1; % Smaller alpha = smoother
torque1_filt = filter(alpha, [1 alpha - 1], torque1);
torque2_filt = filter(alpha, [1 alpha - 1], torque2);

% Plot motor angles
figure;
subplot(3,1,1);
plot(time, rad2deg(theta1_q), 'b', time, rad2deg(theta2_q), 'r');
ylabel('Motor Angles (deg)');
legend('Motor 1', 'Motor 2');
title('Quantized Motor Commands');

% Plot torque signals
subplot(3,1,2);
plot(time, torque1, 'b', time, torque2, 'r');
ylabel('Torque (Nm)');
legend('Motor 1', 'Motor 2');
title('Torque from Acceleration + Damping');

% Plot filtered torque
subplot(3,1,3);
plot(time, torque1_filt, 'b', time, torque2_filt, 'r');
ylabel('Filtered Torque (Nm)');
xlabel('Time (s)');
legend('Motor 1', 'Motor 2');
title('Filtered Torque (Approximated External Load)');

%% BEGINNING AND END VISUALIZATION
% Plot leg at start
figure(1)
disp('Showing starting leg configuration...');
plot_leg_configuration_v2(theta1_q(1), theta2_q(1));

% Pause so user can view
pause(2); % optional

figure(2)
% Plot leg at end
disp('Showing ending leg configuration...');
plot_leg_configuration_v2(theta1_q(end), theta2_q(end));



function [theta1_q, theta2_q, actual_tip, joints, ideal_thetas] = ik_with_motor_resolution(x_target, y_target, resolution_deg)
    res_rad = deg2rad(resolution_deg);

    % Multiple guesses to explore search space
    guesses = [ ...
         pi/3,   pi - pi/3;
        -pi/3,   pi + pi/3;
         pi/6,   pi - pi/6;
        -pi/6,   pi + pi/6;
         0,      pi;
    ];

    % FK constants used for scoring
    L_support = 0.10;

    function [err, tip_y] = ik_obj(thetas)
        try
            [tx, ty, joints_tmp] = robot_leg_fk(thetas(1), thetas(2));
            err = norm([tx; ty] - [x_target; y_target]);

            % Use tip Y-height as proxy for elbow-up preference
            tip_y = ty;
        catch
            err = 1e6;
            tip_y = -Inf;
        end
    end

    % Loop through guesses, prefer lowest error and elbow-up
    best_err = inf;
    best_sol = [NaN; NaN];
    best_tip_y = -Inf;
    opts = optimset('Display', 'off');

    for i = 1:size(guesses, 1)
        [sol, err] = fminsearch(@(th) ik_obj(th), guesses(i,:)', opts);
        [this_err, this_tip_y] = ik_obj(sol);

        % Selection: Prefer lower error, but break ties with higher tip_y (elbow-up)
        if this_err < best_err || (abs(this_err - best_err) < 1e-5 && this_tip_y > best_tip_y)
            best_err = this_err;
            best_sol = sol;
            best_tip_y = this_tip_y;
        end
    end

    % Final unquantized result
    theta1 = best_sol(1);
    theta2 = best_sol(2);
    ideal_thetas = best_sol;

    % Quantize to motor resolution
    theta1_q = round(theta1 / res_rad) * res_rad;
    theta2_q = round(theta2 / res_rad) * res_rad;

    % Use FK with quantized angles
    [tx_q, ty_q, joints] = robot_leg_fk(theta1_q, theta2_q);
    actual_tip = [tx_q; ty_q];
end


function [tip_x, tip_y, joint_positions] = robot_leg_fk(theta1, theta2)
    % Input angles in radians
    % Constants (in meters)
    L_support = 0.10;     % Length of support arm
    L_main = 0.29;        % Total main link length
    L_secondary = 0.20;   % Secondary link length
    offset_from_tip = 0.09; % Distance from tip upward to attach point

    % Motor mount points (same origin)
    baseA = [0; 0];
    baseB = [0; 0];

    % Joint positions at end of each support arm
    % Apply counterrotation + 180° offset to motor B
    jointA = baseA + L_support * [cos(theta1); sin(theta1)];
    jointB = baseB + L_support * [cos(pi - theta2); sin(pi - theta2)];


    % Now solve for tip point P such that:
    % - ||P - jointA|| = L_main
    % - ||jointB - (P - offset_from_tip * unit_vector(P - jointA))|| = L_secondary

    fk_obj = @(tip) abs(norm(tip - jointA) - L_main) + ...
        abs(norm(jointB - (tip - offset_from_tip * (tip - jointA)/norm(tip - jointA))) - L_secondary);

    tip0 = jointA + [0; -L_main];  % initial guess
    tip = fminsearch(fk_obj, tip0);

    % Compute secondary attach point
    unit_vec = (tip - jointA)/norm(tip - jointA);
    joint_secondary = tip - offset_from_tip * unit_vec;

    % Output
    tip_x = tip(1);
    tip_y = tip(2);
    joint_positions = struct('motorA', baseA, 'jointA', jointA, ...
                             'motorB', baseB, 'jointB', jointB, ...
                             'tip', tip, ...
                             'joint_secondary', joint_secondary);
end

function plot_leg_configuration_v2(theta1, theta2)
    % Call FK function
    [tip_x, tip_y, joints] = robot_leg_fk(theta1, theta2);

    % Extract joint coordinates
    A  = joints.motorA;
    JA = joints.jointA;
    B  = joints.motorB;
    JB = joints.jointB;
    P  = joints.tip;
    PS = joints.joint_secondary;  % Attach point on main link

    % Set up plot
    clf; hold on; axis equal;
    grid on;
    xlabel('X (m)'); ylabel('Y (m)');
    title(sprintf('legviz: θa = %.1f°, θb = %.1f°', rad2deg(theta1), rad2deg(theta2)));

    % Plot motor supports
    plot_link(A, JA, 'b', 'arm, motor a');    % Motor A arm
    plot_link(B, JB, 'r', 'arm, motor b');    % Motor B arm

    % Plot links
    plot_link(JA, P, 'k', 'main link');              % Main link
    plot_link(JB, PS, 'm', 'secondary link');        % Secondary link
    plot_link(PS, P, [0.5 0.5 0.5], 'Lower Segment');% Lower segment of main link

    % Mark joints
    plot_joint(A, 'A');
    plot_joint(JA, 'JA');
    plot_joint(B, 'B');
    plot_joint(JB, 'JB');
    plot_joint(P, 'Tip');
    plot_joint(PS, 'PS');


    % Axis orientation: base is higher than tip
    all_y = [A(2), JA(2), B(2), JB(2), PS(2), P(2)];
    all_x = [A(1), JA(1), B(1), JB(1), PS(1), P(1)];
    margin = 0.05;
    xlim([min(all_x)-margin, max(all_x)+margin]);
    ylim([min(all_y)-margin, max(all_y)+margin]);

end

function plot_link(p1, p2, color, label)
    % Plot a link with given color
    plot([p1(1), p2(1)], [p1(2), p2(2)], '-', 'Color', color, 'LineWidth', 3);
    mid = (p1 + p2)/2;
    text(mid(1), mid(2), ['\leftarrow ' label], 'FontSize', 8, 'Color', color);
end

function plot_joint(pt, label)
    % Draw a joint with label
    plot(pt(1), pt(2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 8);
    text(pt(1)+0.01, pt(2), label, 'FontWeight', 'bold');
end