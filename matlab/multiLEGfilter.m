% Parameter Identification & Validation for Robotic Leg
% -------------------------------------------------------------------------
% This script uses your logged runs to:
% 1) Estimate per-motor inertial I and viscous damping b using out-of-substrate
%    runs (least-squares, stacked across runs).
% 2) Compute parameter covariance and standard errors.
% 3) Cross-validate using leave-one-run-out across the out-of-substrate runs.
% 4) Apply the identified model to in-substrate runs to estimate contact torques
%    and convert those to tip forces using a numerical Jacobian of your FK.
%
% REQUIREMENTS / ASSUMPTIONS:
% - You have several CSV files for each run. Each CSV contains columns:
%   time (s), theta1_deg, theta2_deg, torque1 (Nm), torque2 (Nm)
% - The user will set the lists `out_files` and `in_files` below.
% - This script relies on your `robot_leg_fk` function in the same path to
%   compute forward kinematics (used for the Jacobian).
% - The derivatives are computed using a small Savitzky-Golay smoothing step
%   if available; otherwise a simple moving average derivative is used.
%
% OUTPUTS:
% - Identified parameters I1,b1 and I2,b2 with standard errors
% - Cross-validation RMSEs
% - Plots of raw/predicted/detrended torques and estimated tip forces
% -------------------------------------------------------------------------

clear; close all; clc;

%% === USER: list your data files here ===
% Provide full or relative paths to your CSV logs. Separate into out-of-substrate
% (``out_files``) used for identification, and in-substrate (``in_files``)
% for validation / force estimation.

out_files = { 'legacytestruns/horizontal_out.csv', 'legacytestruns/vertical_out.csv', 'legacytestruns/diag_out.csv' };
in_files  = { 'legacytestruns/horizontal_in.csv', 'legacytestruns/vertical_in.csv', 'legacytestruns/diag_in.csv' };

% If your filenames are different, update the lists above.


%% === Load out-of-substrate runs and build regression matrices ===
% We'll fit separate scalar models for each motor:
%   tau1 = I1 * ddth1 + b1 * dth1
%   tau2 = I2 * ddth2 + b2 * dth2

% Stacked regressors and outputs
Phi1_all = []; y1_all = [];
Phi2_all = []; y2_all = [];

% Keep per-run structures for CV
run_data_out = struct();

for i = 1:numel(out_files)
    fname = out_files{i};
    T = readtable(fname);

    % assume columns: time, theta1_deg, theta2_deg, torque1, torque2
    t = T{:,1};
    theta1 = deg2rad(T{:,2});
    theta2 = deg2rad(T{:,3});
    tau1 = T{:,4};
    tau2 = T{:,5};

    dt = mean(diff(t));

    % compute smoothed derivatives
    [d1, dd1] = derivs_smoothed(theta1, dt);
    [d2, dd2] = derivs_smoothed(theta2, dt);

    % Build regressors for this run: [dd, d]
    Phi1 = [dd1, d1];   % Nx2
    Phi2 = [dd2, d2];

    % append
    Phi1_all = [Phi1_all; Phi1];
    y1_all   = [y1_all; tau1];
    Phi2_all = [Phi2_all; Phi2];
    y2_all   = [y2_all; tau2];

    % store for CV
    run_data_out(i).fname = fname;
    run_data_out(i).t = t;
    run_data_out(i).theta1 = theta1;
    run_data_out(i).theta2 = theta2;
    run_data_out(i).d1 = d1; run_data_out(i).dd1 = dd1;
    run_data_out(i).d2 = d2; run_data_out(i).dd2 = dd2;
    run_data_out(i).tau1 = tau1; run_data_out(i).tau2 = tau2;
    run_data_out(i).Phi1 = Phi1; run_data_out(i).Phi2 = Phi2;
end

%% === Least-squares estimate (stacked) ===
% Solve for parameters p1 = [I1; b1], p2 = [I2; b2]
p1_hat = (Phi1_all' * Phi1_all) \ (Phi1_all' * y1_all);
p2_hat = (Phi2_all' * Phi2_all) \ (Phi2_all' * y2_all);

% Compute residuals and sigma^2
res1 = y1_all - Phi1_all * p1_hat;
res2 = y2_all - Phi2_all * p2_hat;

N1 = size(Phi1_all,1); k = size(Phi1_all,2);
sigma2_1 = sum(res1.^2) / (N1 - k);
N2 = size(Phi2_all,1);
sigma2_2 = sum(res2.^2) / (N2 - k);

% Parameter covariance and standard errors
cov_p1 = sigma2_1 * inv(Phi1_all' * Phi1_all);
cov_p2 = sigma2_2 * inv(Phi2_all' * Phi2_all);
se_p1 = sqrt(diag(cov_p1));
se_p2 = sqrt(diag(cov_p2));

fprintf('\nIdentified parameters (stacked out-of-substrate runs):\n');
fprintf(' Motor1: I1 = %.5g ± %.2g (rad^2*Nm*s?),  b1 = %.5g ± %.2g\n', p1_hat(1), se_p1(1), p1_hat(2), se_p1(2));
fprintf(' Motor2: I2 = %.5g ± %.2g, b2 = %.5g ± %.2g\n', p2_hat(1), se_p2(1), p2_hat(2), se_p2(2));

%% === VISUALIZATION: per-run params, torque overlays, component torques ===
% Compute per-run parameter estimates (solve per-out-run) and per-run cov/SE
nRuns = numel(run_data_out);
p1_runs = zeros(nRuns,2); se1_runs = zeros(nRuns,2);
p2_runs = zeros(nRuns,2); se2_runs = zeros(nRuns,2);
rmse_run1 = zeros(nRuns,1); rmse_run2 = zeros(nRuns,1);

for i = 1:nRuns
    Phi1 = run_data_out(i).Phi1; y1 = run_data_out(i).tau1;
    Phi2 = run_data_out(i).Phi2; y2 = run_data_out(i).tau2;
    p1r = (Phi1' * Phi1) \ (Phi1' * y1);
    p2r = (Phi2' * Phi2) \ (Phi2' * y2);
    res1r = y1 - Phi1 * p1r; res2r = y2 - Phi2 * p2r;
    N1r = size(Phi1,1); k = size(Phi1,2);
    s2_1r = sum(res1r.^2) / max(1, N1r - k);
    s2_2r = sum(res2r.^2) / max(1, N1r - k);
    cov1r = s2_1r * inv(Phi1' * Phi1); cov2r = s2_2r * inv(Phi2' * Phi2);
    p1_runs(i,:) = p1r(:)'; se1_runs(i,:) = sqrt(diag(cov1r))';
    p2_runs(i,:) = p2r(:)'; se2_runs(i,:) = sqrt(diag(cov2r))';
    rmse_run1(i) = sqrt(mean(res1r.^2)); rmse_run2(i) = sqrt(mean(res2r.^2));
end

% Plot bar chart of per-run I and b with errorbars
figure('Name','Per-run parameter estimates'); clf;
subplot(2,1,1);
hold on;
bar(1:nRuns, p1_runs(:,1)); errorbar(1:nRuns, p1_runs(:,1), se1_runs(:,1), '.k');
xlabel('Out run'); ylabel('I_1'); title('Motor1: Inertial parameter per run ± SE');
subplot(2,1,2);
hold on;
bar(1:nRuns, p1_runs(:,2)); errorbar(1:nRuns, p1_runs(:,2), se1_runs(:,2), '.k');
xlabel('Out run'); ylabel('b_1'); title('Motor1: Viscous b per run ± SE');

figure('Name','Per-run parameter estimates Motor2'); clf;
subplot(2,1,1);
hold on; bar(1:nRuns, p2_runs(:,1)); errorbar(1:nRuns, p2_runs(:,1), se2_runs(:,1), '.k');
xlabel('Out run'); ylabel('I_2'); title('Motor2: Inertial parameter per run ± SE');
subplot(2,1,2);
hold on; bar(1:nRuns, p2_runs(:,2)); errorbar(1:nRuns, p2_runs(:,2), se2_runs(:,2), '.k');
xlabel('Out run'); ylabel('b_2'); title('Motor2: Viscous b per run ± SE');

% Compare raw vs predicted torque for each out-of-substrate run
figure('Name','Out-of-substrate: raw vs per-run and stacked predictions'); clf;
for i = 1:nRuns
    t = run_data_out(i).t;
    tau1 = run_data_out(i).tau1; d1 = run_data_out(i).d1; dd1 = run_data_out(i).dd1;
    tau2 = run_data_out(i).tau2; d2 = run_data_out(i).d2; dd2 = run_data_out(i).dd2;

    % per-run preds
    p1r = p1_runs(i,:)'; p2r = p2_runs(i,:)';
    tau1_pr = p1r(1).*dd1 + p1r(2).*d1;
    tau2_pr = p2r(1).*dd2 + p2r(2).*d2;

    % stacked preds
    tau1_st = p1_hat(1).*dd1 + p1_hat(2).*d1;
    tau2_st = p2_hat(1).*dd2 + p2_hat(2).*d2;

    ax = subplot(nRuns,2,2*(i-1)+1);
    plot(t, tau1, 'k', t, tau1_pr, '--b', t, tau1_st, ':c'); hold on;
    xlabel('t (s)'); ylabel('Torque1 (Nm)');
    title(sprintf('Run %d: Motor1 (raw, per-run, stacked)', i));
    legend('raw','per-run','stacked');

    ax2 = subplot(nRuns,2,2*(i-1)+2);
    plot(t, tau2, 'k', t, tau2_pr, '--r', t, tau2_st, ':m'); hold on;
    xlabel('t (s)'); ylabel('Torque2 (Nm)');
    title(sprintf('Run %d: Motor2 (raw, per-run, stacked)', i));
    legend('raw','per-run','stacked');
end

% Plot separate torque components (inertia vs damping) for stacked fit (for each out run)
figure('Name','Torque components (stacked)'); clf;
for i = 1:nRuns
    t = run_data_out(i).t;
    d1 = run_data_out(i).d1; dd1 = run_data_out(i).dd1;
    d2 = run_data_out(i).d2; dd2 = run_data_out(i).dd2;

    I1comp = p1_hat(1).*dd1;
    B1comp = p1_hat(2).*d1;
    I2comp = p2_hat(1).*dd2;
    B2comp = p2_hat(2).*d2;

    subplot(nRuns,2,2*(i-1)+1);
    plot(t, I1comp, '--b', t, B1comp, ':b', t, I1comp + B1comp, '-k'); hold on;
    xlabel('t'); ylabel('Torque (Nm)');
    title(sprintf('Run %d Motor1 components (I*dd, b*d, total)', i));
    legend('I*dd','b*d','total');

    subplot(nRuns,2,2*(i-1)+2);
    plot(t, I2comp, '--r', t, B2comp, ':r', t, I2comp + B2comp, '-k'); hold on;
    xlabel('t'); ylabel('Torque (Nm)');
    title(sprintf('Run %d Motor2 components (I*dd, b*d, total)', i));
    legend('I*dd','b*d','total');
end

% Print a compact table with stacked vs mean(per-run) and coefficient of variation
fprintf('\nStacked vs per-run summary:\n');
fprintf('  Motor1: stacked I=%.4g, mean(per-run I)=%.4g, cv(I)=%.2f\n', p1_hat(1), mean(p1_runs(:,1)), std(p1_runs(:,1))/abs(mean(p1_runs(:,1))));
fprintf('          stacked b=%.4g, mean(per-run b)=%.4g, cv(b)=%.2f\n', p1_hat(2), mean(p1_runs(:,2)), std(p1_runs(:,2))/abs(mean(p1_runs(:,2))));
fprintf('  Motor2: stacked I=%.4g, mean(per-run I)=%.4g, cv(I)=%.2f\n', p2_hat(1), mean(p2_runs(:,1)), std(p2_runs(:,1))/abs(mean(p2_runs(:,1))));
fprintf('          stacked b=%.4g, mean(per-run b)=%.4g, cv(b)=%.2f\n', p2_hat(2), mean(p2_runs(:,2)), std(p2_runs(:,2))/abs(mean(p2_runs(:,2))));


%% === Cross-validation: leave-one-run-out ===
nRuns = numel(run_data_out);
rmse_cv_1 = zeros(nRuns,1);
rmse_cv_2 = zeros(nRuns,1);

for i = 1:nRuns
    % build training by excluding i
    Phi1_train = []; y1_train = [];
    Phi2_train = []; y2_train = [];
    for j = 1:nRuns
        if j==i, continue; end
        Phi1_train = [Phi1_train; run_data_out(j).Phi1];
        y1_train = [y1_train; run_data_out(j).tau1];
        Phi2_train = [Phi2_train; run_data_out(j).Phi2];
        y2_train = [y2_train; run_data_out(j).tau2];
    end
    p1_cv = (Phi1_train' * Phi1_train) \ (Phi1_train' * y1_train);
    p2_cv = (Phi2_train' * Phi2_train) \ (Phi2_train' * y2_train);

    % test on left-out run i
    tau1_pred = run_data_out(i).Phi1 * p1_cv;
    tau2_pred = run_data_out(i).Phi2 * p2_cv;
    rmse_cv_1(i) = sqrt(mean((run_data_out(i).tau1 - tau1_pred).^2));
    rmse_cv_2(i) = sqrt(mean((run_data_out(i).tau2 - tau2_pred).^2));
end

fprintf('\nCross-validation RMSE (leave-one-run-out) per run:\n');
for i=1:nRuns
    fprintf(' Run %d (%s): RMSE1=%.4e, RMSE2=%.4e\n', i, run_data_out(i).fname, rmse_cv_1(i), rmse_cv_2(i));
end

%% === Apply identified model to in-substrate runs and estimate tip forces ===
figure('Name','In-Substrate Results');
for i = 1:numel(in_files)
    fname = in_files{i};
    T = readtable(fname);
    t = T{:,1};
    theta1 = deg2rad(T{:,2});
    theta2 = deg2rad(T{:,3});
    tau1 = T{:,4}; tau2 = T{:,5};
    dt = mean(diff(t));

    % get smoothed derivatives
    [d1, dd1] = derivs_smoothed(theta1, dt);
    [d2, dd2] = derivs_smoothed(theta2, dt);

    % predicted torque from identified params
    tau1_pred = p1_hat(1) * dd1 + p1_hat(2) * d1;
    tau2_pred = p2_hat(1) * dd2 + p2_hat(2) * d2;

    % detrended torque (contact + residuals)
    tau1_contact = tau1 - tau1_pred;
    tau2_contact = tau2 - tau2_pred;

    % convert to tip force F = (J^T)^(-1) * tau (numerical Jacobian per sample)
    n = length(t);
    Fx = zeros(n,1); Fy = zeros(n,1);
    for k = 1:n
        th1 = theta1(k); th2 = theta2(k);
        J = numeric_jacobian(th1, th2);
        % solve J' * F = tau_contact  -> F = (J') \ tau_contact
        tau_vec = [tau1_contact(k); tau2_contact(k)];
        % guard against singular J
        if abs(det(J)) < 1e-8
            Fx(k) = NaN; Fy(k) = NaN;
        else
            F = (J') \ tau_vec;
            Fx(k) = F(1); Fy(k) = F(2);
        end
    end

    % Plot torques and forces for this run
    subplot(numel(in_files),2,2*(i-1)+1);
    plot(t, tau1, 'b', t, tau1_pred, '--b', t, tau1_contact, ':b'); hold on;
    plot(t, tau2, 'r', t, tau2_pred, '--r', t, tau2_contact, ':r');
    xlabel('Time (s)'); ylabel('Torque (Nm)');
    title(sprintf('Run: %s — Torque (raw, pred, contact)', fname));
    legend('raw1','pred1','cont1','raw2','pred2','cont2');

    subplot(numel(in_files),2,2*(i-1)+2);
    plot(t, Fx, 'k', t, Fy, 'g');
    xlabel('Time (s)'); ylabel('Tip Force (N)');
    title(sprintf('Run: %s — Estimated Tip Force', fname));
    legend('Fx','Fy');
end

%% === Save identified parameters for later use ===
save('identified_params.mat','p1_hat','p2_hat','cov_p1','cov_p2','se_p1','se_p2');

fprintf('\nSaved identified_params.mat with p1_hat and p2_hat.\n');

%% === End of script ===

% Notes:
% - After running, inspect residuals (res1, res2) and RMSEs. If residuals are
%   large or show structure vs theta, consider configuration-dependent models.
% - The Jacobian used is numerical. For more robust force estimation, consider
%   deriving an analytic Jacobian for the mechanism (if available).

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
    jointB = baseB + L_support * [cos(theta2); sin(theta2)];


    % Now solve for tip point P such that:
    % - ||P - jointA|| = L_main
    % - ||jointB - (P - offset_from_tip * unit_vector(P - jointA))|| = L_secondary

    fk_obj = @(tip) abs(norm(tip - jointA) - L_main) + ...
        abs(norm(jointB - (tip - offset_from_tip * (tip - jointA)/norm(tip - jointA))) - L_secondary);

    tip0 = jointA + [0; -L_main];  % initial guess
    options = optimset('Display','off');
    tip = fminsearch(fk_obj, tip0,options);

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

%% === Helper functions (nested) ===
% small utility: numerical Jacobian of robot_leg_fk at (th1, th2)
function J = numeric_jacobian(th1, th2)
    eps_ang = 1e-6; % rad
    [x0, y0, ~] = robot_leg_fk(th1, th2);
    [x1, y1, ~] = robot_leg_fk(th1 + eps_ang, th2);
    [x2, y2, ~] = robot_leg_fk(th1, th2 + eps_ang);
    dx_dth1 = (x1 - x0) / eps_ang; dy_dth1 = (y1 - y0) / eps_ang;
    dx_dth2 = (x2 - x0) / eps_ang; dy_dth2 = (y2 - y0) / eps_ang;
    J = [dx_dth1, dx_dth2; dy_dth1, dy_dth2];
end

% derivative helper: compute smoothed derivatives for a vector sampled at dt
function [xdot, xddot] = derivs_smoothed(x, dt)
    N = length(x);
    % Try to use Savitzky-Golay if available; if not, fall back to moving-average
    window = max(5, 2*floor(0.05*N/2)+1); % ~5% of length, odd
    polyorder = 3;
    if exist('sgolayfilt','file')
        xs = sgolayfilt(x, polyorder, window);
        xdot = [0; diff(xs)/dt];
        xddot = [0; diff(xdot)/dt];
    else
        % fallback: moving average smoothing then finite difference
        xs = movmean(x, window);
        xdot = [0; diff(xs)/dt];
        xddot = [0; diff(xdot)/dt];
    end
end
