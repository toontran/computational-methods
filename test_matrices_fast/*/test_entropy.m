%% Check preserved normalized collision entropy when rows double
% Old response: y = N_{w-1} u  in R^m
% New response: z = A_{I_w} u  in R^m
% Combined:     [y; z]         in R^(2m)

clear; clc;

%% Example data
% Try different choices here
m = 8;

% Example 1: nearly concentrated old response
y = [1; 0.15; 0; 0; 0; 0; 0; 0];

% Example 2: maximally entropic old response
% y = 0.35 * ones(m,1);

% New rows to test
z = [0.4; 0; 0; 0; 0; 0; 0; 0];

%% Helper functions
P = @(x) sum(abs(x).^2);
Q = @(x) sum(abs(x).^4);

% Collision entropy of normalized vector x / ||x||_2
H2 = @(x) -log(Q(x) / (P(x)^2));

% Normalized entropy: divide by max possible entropy log(length(x))
Hnorm = @(x) H2(x) / log(length(x));

%% Compute quantities
yc = [y; z];

P_old = P(y);
Q_old = Q(y);
H_old = H2(y);
Hn_old = Hnorm(y);

P_new = P(yc);
Q_new = Q(yc);
H_new = H2(yc);
Hn_new = Hnorm(yc);

alpha = log(2*m) / log(m);

lhs_ratio = Q_new / (P_new^2);
rhs_ratio = (Q_old / (P_old^2))^alpha;

fprintf('m = %d\n\n', m);

fprintf('Old block:\n');
fprintf('  P_old   = %.12g\n', P_old);
fprintf('  Q_old   = %.12g\n', Q_old);
fprintf('  H_old   = %.12g\n', H_old);
fprintf('  Hn_old  = %.12g\n\n', Hn_old);

fprintf('Combined block [y; z]:\n');
fprintf('  P_new   = %.12g\n', P_new);
fprintf('  Q_new   = %.12g\n', Q_new);
fprintf('  H_new   = %.12g\n', H_new);
fprintf('  Hn_new  = %.12g\n\n', Hn_new);

fprintf('Check preserved normalized entropy:\n');
fprintf('  Hn_new - Hn_old = %.12g\n\n', Hn_new - Hn_old);

fprintf('Equivalent ratio check:\n');
fprintf('  Q_new / P_new^2                = %.12g\n', lhs_ratio);
fprintf('  (Q_old / P_old^2)^(log(2m)/log(m)) = %.12g\n', rhs_ratio);
fprintf('  difference                     = %.12g\n\n', lhs_ratio - rhs_ratio);

%% Solve for the exact q_new required by the theory, given p_new = ||z||_2^2
p_new = P(z);

q_new_required = (Q_old / P_old^2)^alpha * (P_old + p_new)^2 - Q_old;
q_new_actual   = Q(z);

fprintf('Given p_new = ||z||_2^2:\n');
fprintf('  q_new_actual   = ||z||_4^4 = %.12g\n', q_new_actual);
fprintf('  q_new_required           = %.12g\n', q_new_required);
fprintf('  difference               = %.12g\n\n', q_new_actual - q_new_required);

%% Optional: construct a constant-magnitude z that matches the required q_new approximately
% For a constant vector z = a * ones(m,1):
%   ||z||_2^2 = m a^2,  ||z||_4^4 = m a^4 = (||z||_2^2)^2 / m
%
% So this only works when q_new_required = p_new^2 / m.

q_uniform_from_p = p_new^2 / m;
fprintf('If z were uniform with same ||z||_2^2:\n');
fprintf('  q_uniform_from_p = p_new^2 / m = %.12g\n', q_uniform_from_p);
fprintf('  compare to q_new_required      = %.12g\n', q_new_required);


%% Random search for z whose normalized entropy is close to preserved
clearvars -except y m P Q H2 Hnorm

num_trials = 5000;
best_err = inf;
best_z = [];
best_vals = struct();

target = Hnorm(y);

for t = 1:num_trials
    z = randn(m,1);

    yc = [y; z];
    err = abs(Hnorm(yc) - target);

    if err < best_err
        best_err = err;
        best_z = z;
        best_vals.Pz = P(z);
        best_vals.Qz = Q(z);
        best_vals.Hn_combined = Hnorm(yc);
    end
end

fprintf('Best random-search error in normalized entropy: %.12g\n', best_err);
fprintf('Best z stats:\n');
fprintf('  ||z||_2^2 = %.12g\n', best_vals.Pz);
fprintf('  ||z||_4^4 = %.12g\n', best_vals.Qz);
fprintf('  combined normalized entropy = %.12g\n', best_vals.Hn_combined);

disp('Best z found:');
disp(best_z);