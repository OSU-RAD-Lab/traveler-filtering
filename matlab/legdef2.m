
% Link lengths
L = [0.2; 0.25];

% Foot force (combined lateral/vertical force)
F_tip = [40; 10];

% Grid of foot positions
x_vals = linspace(-0.4, 0.4, 25);
y_vals = linspace(-0.8, 0, 25);
[X, Y] = meshgrid(x_vals, y_vals);

% Allocate torque matrices
T1_map = NaN(size(X));  % Joint 1 torques
T2_map = NaN(size(X));  % Joint 2 torques
z = 0;
% Loop over each point in the workspace
for i = 1:numel(X)
    pos = [X(i); Y(i)];
    try
        torques = hanging_leg_ik_force(pos, F_tip, L);
        T1_map(i) = torques(1);
        T2_map(i) = torques(2);
    catch
        % Point is unreachable; leave as NaN
    end
    z = z+1
end


%% PLOT CONTROLS


% Plot Joint 1 torque heatmap
figure(1); clf;
hold on
t1map = surf(X, Y, T1_map, 'EdgeColor', 'none');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Torque Joint 1 (Nm)');
title(sprintf('Joint 1 Torque Across Workspace, Tip Load: [%.0f, %.0f] N [X,Y]',F_tip(1),F_tip(2)));
colorbar;
shading interp;
t1zone = zone_draw(.3,.2,-.25);
view(2);  % Top-down heatmap view
uistack(t1zone,'top')
axis equal
hold off

% Plot Joint 2 torque heatmap
figure(2); clf;
hold on
t2map = surf(X, Y, T2_map, 'EdgeColor','none');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Torque Joint 2 (Nm)');
title(sprintf('Joint 2 Torque Across Workspace, Tip Load: [%.0f, %.0f] N [X,Y]',F_tip(1),F_tip(2)));
colorbar;
shading interp;
t2zone = zone_draw(.3,.2,-.25);
view(2);  % Top-down heatmap view
uistack(t2zone,'top')
axis equal
hold off

Tmap = sqrt(T1_map.^2 + T2_map.^2);

% Plot Combined torque heatmap
figure(3); clf;
hold on
t2map = surf(X, Y, Tmap, 'EdgeColor','none');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Torque Magnitude (Nm)');
title(sprintf('Combined Torque Across Workspace, Tip Load: [%.0f, %.0f] N [X,Y]',F_tip(1),F_tip(2)));
colorbar;
shading interp;
t2zone = zone_draw(.3,.2,-.25);
view(2);  % Top-down heatmap view
uistack(t2zone,'top')
axis equal
hold off




function joint_torques = hanging_leg_ik_force(xy_tip, F_tip, L)
    % xy_tip: [x; y] desired foot position (global)
    % F_tip: [Fx; Fy] desired force at the foot
    % L: [L1; L2] link lengths

    x = xy_tip(1);
    y = xy_tip(2);
    L1 = L(1);
    L2 = L(2);

    %--- Inverse Kinematics
    D = (x^2 + y^2 - L1^2 - L2^2) / (2 * L1 * L2);
    if abs(D) > 1
        error('Target is unreachable with given link lengths.');
    end

    theta2 = atan2(-sqrt(1 - D^2), D); % "knee down" configuration
    k1 = L1 + L2*cos(theta2);
    k2 = L2*sin(theta2);
    theta1 = atan2(y, x) - atan2(k2, k1);

    theta = [theta1; theta2];

    %--- Kinematics
    hip = [0; 0];
    knee = hip + L1 * [cos(theta1); sin(theta1)];
    foot = knee + L2 * [cos(theta1 + theta2); sin(theta1 + theta2)];

    %--- Jacobian
    J = [ -L1*sin(theta1) - L2*sin(theta1 + theta2), -L2*sin(theta1 + theta2);
           L1*cos(theta1) + L2*cos(theta1 + theta2),  L2*cos(theta1 + theta2) ];

    joint_torques = J' * F_tip;

    %--- Visualizer
    % figure(1); clf; hold on; axis equal;
    % plot([hip(1), knee(1)], [hip(2), knee(2)], 'b-', 'LineWidth', 4);
    % plot([knee(1), foot(1)], [knee(2), foot(2)], 'r-', 'LineWidth', 4);
    % plot(foot(1), foot(2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    % quiver(foot(1), foot(2), F_tip(1)/5, F_tip(2)/5, 0, 'k', 'LineWidth', 2);
    % 
    % xlabel('X'); ylabel('Y');
    % title(sprintf('Foot @ [%.2f, %.2f] | θ1 = %.1f°, θ2 = %.1f°', ...
    %     x, y, rad2deg(theta1), rad2deg(theta2)));
    % grid on;
    % xlim([-sum(L), sum(L)]); ylim([-sum(L), sum(L)]);
end

function out = zone_draw(w,l,v)
z = w/2;
b = l/2;
c = 100;
% upper corner:
x1 = -z;
y1 = -b+v;

% lower corner:

x2 = z;
y2 = b+v;

%plot 
x = [x1 x2 x2 x1 x1];
y = [y1 y1 y2 y2 y1];
z = [c,c,c,c,c];
out = plot3(x, y, z, 'r-', 'LineWidth', 2);
end
