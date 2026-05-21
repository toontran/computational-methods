%% Streaming iSVD / FD / EntropyScore experiments over varying sigma1
% EntropyScore now supports subsequent row blocks using carried old-state:
%   s_j^2  ~= ||N_old v_j||_2^2
%   q_j    ~= ||N_old v_j||_4^4 = s_j^4 * exp(-H_j)
%
% For block 1:
%   optimize exact entropy score on A_1.
%
% For block w >= 2:
%   optimize approximate score
%     log score(v)
%       = log ||M v||_2
%         - gamma * H_approx(v),
%   where
%     M = [B; A_w],  B = S_old * V_old',
%     gamma = log(rows_total / n) / (2 log(rows_total)),
%     H_approx(v) = -log( Q(v) / E(v)^2 ),
%     E(v) = sum_j a_j^2 s_j^2 + ||A_w v||_2^2,
%     Q(v) = sum_j a_j^4 q_j   + ||A_w v||_4^4,
%     a = V_old' * v.
%
% After extracting each vector v, the carried state is updated via
%     s_new(v)^2 = sum_j a_j^2 s_j^2 + ||A_w v||_2^2,
%     q_new(v)   = sum_j a_j^4 q_j   + ||A_w v||_4^4.
%
% This script is self-contained.

clear; clc;
rng(0);

tic
%% Basic parameters
n    = 1024;
r    = 1;               % target rank to keep in streaming
l    = 1;               % # true right singular vectors to track (must satisfy l <= r)
win  = 100;             % window / block size
mode = "EntropyScore";  % "iSVD", "FD", or "EntropyScore"
V_type = "id";          % "id", "U", or "rand"
r_sig = 1;              % true signal-block rank used in U,S construction

format compact;
alpha_sig  = 0.003;
alpha_tail = 0.0145;
tail_scale = 0.99;

coarse_svals = [0.991];
first_svals  = coarse_svals;

num_svals = numel(first_svals);
num_exper = 1;

%% --- Consistent low-rank ground truth setup ---
k = n;

U0 = zeros(n, n);

if mod(n,2) ~= 0
    error('Hadamard construction requires n to be even (typically a power of 2).');
end
H = hadamard(n);
U0(:, 1:r_sig) = H(:, 1:r_sig) / sqrt(n);

a_tail = sqrt(1 - r_sig/n);
b_tail = 1/sqrt(n);
for j = r_sig+1:n
    col = zeros(n,1);
    idx_large = j - r_sig;
    if idx_large <= n - r_sig
        col(idx_large) = a_tail;
    else
        error('Tail index out of range; reduce r_sig or adjust construction.');
    end
    col(n-r_sig+1:n) = b_tail;
    U0(:, j) = col;
end

[Qtmp, ~] = qr(U0, 0);
for j = 1:r_sig
    if dot(Qtmp(:, j), U0(:, j)) < 0
        Qtmp(:, j) = -Qtmp(:, j);
    end
end
U = Qtmp(:, 1:k);

switch V_type
    case "id"
        V = eye(n, k);
    case "U"
        V = U;
    case "rand"
        [V, ~] = qr(randn(n, k), 0);
    otherwise
        error('Unknown V_type. Use "id", "U", or "rand".');
end

%% Allocate storage for results
alignment_results   = zeros(num_svals, num_exper);
relerr_sval_results = zeros(num_svals, num_exper);
Delta_results       = zeros(num_svals, num_exper);
DeltaComp_results   = zeros(num_svals, num_exper);
low_sval_indicator  = zeros(num_svals, num_exper);

%% Outer loop over sigma1 values
for i = 1:num_svals
    sigma1 = first_svals(i);

    sig_block  = sigma1 * (1:r_sig).^(-alpha_sig);
    tail_block = tail_scale * (1:(k-r_sig)).^(-alpha_tail);

    svec = [sig_block, tail_block];
    svec(1) = sigma1;
    S = diag(svec);

    if r < k
        E_opt = sum(svec(r+1:end).^2);
    else
        E_opt = 0;
    end

    Delta_comp = sum(svec(1:r).^2) - sum(svec(r+1:2*r).^2);
    DeltaComp_results(i, :) = Delta_comp;

    for e = 1:num_exper
        p = randperm(n);
        A = U * S * V';
        A = A(p, :);

        [mA, ~] = size(A);

        % Streaming state
        state = [];
        V_r = [];
        S_r = [];
        H_r = [];
        score_r = [];

        % Streaming over row blocks
        for start_row = 1:win:mA
            end_row = min(start_row + win - 1, mA);
            A_block = A(start_row:end_row, :);

            switch mode
                case "EntropyScore"
                    if isempty(state)
                        fprintf('\n===== block rows %d:%d (initial exact block) =====\n', start_row, end_row);
                    else
                        fprintf('\n===== block rows %d:%d (streaming approximate block) =====\n', start_row, end_row);
                    end

                    [V_new, s_new, H_new, score_new, state_new] = ...
                        entropy_iter_basis_streaming( ...
                            A_block, r, n, state, V_r, ...
                            8, ...      % num_restarts
                            200, ...    % maxit
                            1e-8);      % tol

                    V_r = V_new;
                    S_r = diag(s_new);
                    H_r = H_new;
                    score_r = score_new;
                    state = state_new;

                    fprintf('rows %d:%d\n', start_row, end_row);
                    fprintf('s: ');      disp(s_new(:)');
                    fprintf('H: ');      disp(H_new(:)');
                    fprintf('scores: '); disp(score_new(:)');

                    if start_row == 1
                        [~, ~, vtmp] = svd(A_block, "econ");
                        e1_proj = vtmp*vtmp' * V(:,1);
                        if norm(e1_proj) > 1e-14
                            score_e1_proj = entropy_score_fast(A_block, e1_proj / norm(e1_proj), size(A_block,1), n);
                            fprintf('score of v1 projection onto window space: '); disp(score_e1_proj);
                            fprintf('actual score: '); disp(score_new);
                            fprintf('V(1,1)=%.5f\n', V_new(1,1));
                            fprintf('should be: %.5f\n', e1_proj(1)/norm(e1_proj));
                        end
                    else
                        if ~isempty(state.prev_basis)
                            a_dbg = state.prev_basis' * V_new(:,1);
                            y_dbg = A_block * V_new(:,1);
                            E_old_dbg = sum((a_dbg.^2) .* state.prev_s2);
                            Q_old_dbg = sum((a_dbg.^4) .* state.prev_q);
                            fprintf('debug E_old(first vec)=%.12e\n', E_old_dbg);
                            fprintf('debug Q_old(first vec)=%.12e\n', Q_old_dbg);
                            fprintf('debug ||A_w v||_2^2(first vec)=%.12e\n', kahan_sum(abs(y_dbg).^2));
                            fprintf('debug ||A_w v||_4^4(first vec)=%.12e\n', kahan_sum(abs(y_dbg).^4));
                        end
                    end

                case {"iSVD", "FD"}
                    if isempty(V_r)
                        M = A_block;
                    else
                        B_top = S_r * V_r';
                        M     = [B_top; A_block];
                    end

                    [U_hat, S_hat, V_hat] = svd(M, 'econ');
                    s = diag(S_hat);
                    rr = min(r, numel(s));

                    switch mode
                        case "iSVD"
                            S_r = S_hat(1:rr, 1:rr);
                            V_r = V_hat(:, 1:rr);

                        case "FD"
                            if numel(s) > rr
                                delta = s(rr+1)^2;
                            else
                                delta = 0;
                            end
                            s1 = s(1:rr);
                            s1_shr = sqrt(max(s1.^2 - delta, 0));
                            S_r = diag(s1_shr);
                            V_r = V_hat(:, 1:rr);
                    end

                otherwise
                    error('Unknown mode. Use "iSVD", "FD", or "EntropyScore".');
            end
        end

        %% --- Metrics after full pass ---
        ll = min(l, size(V_r,2));
        align = norm((eye(size(V_r,1)) - V_r*V_r') * V(:,1:ll), 'fro') / sqrt(ll);

        if isempty(S_r)
            top_sval_est = 0;
        else
            top_sval_est = S_r(1,1);
        end

        rel_err_sval = abs(top_sval_est - sigma1) / sigma1;

        if isempty(V_r)
            E_alg = norm(A, 'fro')^2;
        else
            E_alg = norm(A - A * V_r * V_r', 'fro')^2;
        end
        Delta = E_alg - E_opt;

        alignment_results(i, e)   = align;
        relerr_sval_results(i, e) = rel_err_sval;
        Delta_results(i, e)       = Delta;
        low_sval_indicator(i, e)  = double(top_sval_est <= 0.99);
    end
end

%% Summaries
mean_align = mean(alignment_results, 2);
std_align  = std(alignment_results, 0, 2);

mean_relerr_sval = mean(relerr_sval_results, 2);
std_relerr_sval  = std(relerr_sval_results, 0, 2);

low_sval_count    = sum(low_sval_indicator, 2);
low_sval_fraction = low_sval_count / num_exper; %#ok<NASGU>

summary_table = table( ...
    first_svals(:), ...
    mean_align, std_align, ...
    mean_relerr_sval, std_relerr_sval, ...
    low_sval_count, ...
    'VariableNames', ...
    {'sigma1', 'mean_align', 'std_align', ...
     'mean_relerr_sval', 'std_relerr_sval', ...
     sprintf('count_sval_le_099_over_%d', num_exper)} ...
);
disp(summary_table);

elapsedTime = toc;
fprintf('Elapsed time: %.3f\n', elapsedTime);

%% ========================= Local helper functions =========================

function [V_out, s_out, H_out, score_out, state_out] = entropy_iter_basis_streaming( ...
    A_block, r, n, state_prev, V_init, num_restarts, maxit, tol)

    d = size(A_block, 2);
    rows_new = size(A_block, 1);

    V_out = zeros(d, r);
    s_out = zeros(r, 1);
    H_out = -inf(r, 1);
    score_out = -inf(r, 1);

    Q = zeros(d, 0);

    is_initial_block = isempty(state_prev);

    if is_initial_block
        rows_total = rows_new;
        M_gain = A_block;
        prev_basis = [];
        prev_s2 = [];
        prev_q = [];
    else
        rows_total = state_prev.rows_seen + rows_new;
        B_top = diag(state_prev.s) * state_prev.V';
        M_gain = [B_top; A_block];
        prev_basis = state_prev.V;
        prev_s2 = state_prev.s2;
        prev_q = state_prev.q;
    end

    for k = 1:r
        starts = make_basic_restart_seeds(M_gain, Q, k, V_init, num_restarts);

        best_v = [];
        best_logf = -inf;
        best_s2 = 0;
        best_H = inf;

        for restart = 1:num_restarts
            v0 = starts{restart};

            if is_initial_block
                [v_cand, logf_cand, s2_cand, H_cand] = ...
                    basic_projected_ascent_single_exact( ...
                        A_block, v0, Q, rows_total, n, maxit, tol);
            else
                [v_cand, logf_cand, s2_cand, H_cand] = ...
                    basic_projected_ascent_single_streaming( ...
                        M_gain, A_block, prev_basis, prev_s2, prev_q, ...
                        rows_total, n, v0, Q, maxit, tol);
            end

            if logf_cand > best_logf
                best_logf = logf_cand;
                best_v = v_cand;
                best_s2 = s2_cand;
                best_H = H_cand;
            end
        end

        if isempty(best_v)
            error('All restarts failed for k=%d.', k);
        end

        Q = [Q, best_v];
        V_out(:, k) = best_v;
        s_out(k) = sqrt(max(best_s2, 0));
        H_out(k) = best_H;
        score_out(k) = exp(best_logf);
    end

    s2_out = s_out.^2;
    q_out = (s_out.^4) .* exp(-H_out);

    state_out = struct();
    state_out.V = V_out;
    state_out.s = s_out;
    state_out.s2 = s2_out;
    state_out.H = H_out;
    state_out.q = q_out;
    state_out.score = score_out;
    state_out.rows_seen = rows_total;

    % Keep previous state too for debugging / inspection
    state_out.prev_basis = prev_basis;
    state_out.prev_s2 = prev_s2;
    state_out.prev_q = prev_q;
end

function starts = make_basic_restart_seeds(M, Q, k, V_init, num_restarts)
    d = size(M,2);
    starts = cell(num_restarts,1);

    [~, ~, Vsvd] = svd(M, "econ");
    num_top = min(4, size(Vsvd, 2));
    alpha_grid = [0.98, 0.9, 0.75, 0.5, 0.25, 0.0];

    for restart = 1:num_restarts
        if ~isempty(V_init) && size(V_init,2) >= k
            v_prev = V_init(:,k);
        else
            v_prev = [];
        end

        restart_type = mod(restart - 1, 5) + 1;
        restart_block = floor((restart - 1) / 5);

        switch restart_type
            case 1
                if ~isempty(v_prev)
                    xi = randn(d,1);
                    xi = project_feasible(xi, Q);
                    nxi = sqrt(kahan_sum(abs(xi).^2));
                    if nxi > 1e-14
                        xi = xi / nxi;
                    end
                    alpha = alpha_grid(mod(restart_block, numel(alpha_grid)) + 1);
                    v0 = alpha * v_prev + sqrt(max(0, 1 - alpha^2)) * xi;
                else
                    v0 = Vsvd(:,1);
                end

            case 2
                j = mod(restart_block, num_top) + 1;
                v0 = Vsvd(:, j);

            case 3
                j1 = mod(restart_block, num_top) + 1;
                j2 = mod(restart_block + 1, num_top) + 1;
                alpha = alpha_grid(mod(restart_block, numel(alpha_grid)) + 1);
                v0 = alpha * Vsvd(:, j1) + sqrt(max(0, 1 - alpha^2)) * Vsvd(:, j2);

            case 4
                j = mod(restart_block, num_top) + 1;
                v0 = Vsvd(:, j) + 1e-2 * randn(d,1);

            otherwise
                v0 = randn(d,1);
        end

        v = retract_feasible(v0, Q);
        if isempty(v)
            v = retract_feasible(randn(d,1), Q);
        end
        if isempty(v)
            error('Could not generate feasible restart seed.');
        end
        starts{restart} = v;
    end
end

function [v, logf, s2, H2] = basic_projected_ascent_single_exact( ...
    M, v0, Q, rows_total, n, maxit, tol)

    v = retract_feasible(v0, Q);
    if isempty(v)
        error('Initial vector infeasible in exact optimizer.');
    end

    [logf, gradE, s2, H2] = entropy_logscore_grad_rows(M, v, rows_total, n);

    progress_f_tol = 1e-12;
    progress_step_tol = 1e-10;

    for it = 1:maxit %#ok<NASGU>
        g = project_to_feasible_tangent(gradE, v, Q);
        gnorm = sqrt(kahan_sum(abs(g).^2));
        if gnorm <= tol
            return;
        end

        accepted = false;
        alpha = 1.0;
        logf_old = logf;
        v_old = v;

        for ls_it = 1:20 %#ok<NASGU>
            vt = retract_feasible(v + alpha * g, Q);
            if ~isempty(vt)
                [logf_trial, ~, ~, ~] = entropy_logscore_grad_rows(M, vt, rows_total, n);
                rhs = logf_old + 1e-4 * alpha * real(g' * g);
                if logf_trial >= rhs
                    accepted = true;
                    v = vt;
                    break;
                end
            end
            alpha = 0.5 * alpha;
        end

        if ~accepted
            v = v_old;
            return;
        end

        [logf, gradE, s2, H2] = entropy_logscore_grad_rows(M, v, rows_total, n);

        step_norm = sqrt(kahan_sum(abs(v - v_old).^2));
        f_change = abs(logf - logf_old);
        f_threshold = progress_f_tol * max(1.0, abs(logf_old));

        if f_change <= f_threshold || step_norm <= progress_step_tol
            return;
        end
    end
end

function [v, logf, s2_total, H_approx] = basic_projected_ascent_single_streaming( ...
    M_gain, A_block, V_old, s2_old, q_old, rows_total, n, v0, Q, maxit, tol)

    v = retract_feasible(v0, Q);
    if isempty(v)
        error('Initial vector infeasible in streaming optimizer.');
    end

    [logf, gradE, s2_total, H_approx] = entropy_streaming_logscore_grad( ...
        M_gain, A_block, V_old, s2_old, q_old, v, rows_total, n);

    progress_f_tol = 1e-12;
    progress_step_tol = 1e-10;

    for it = 1:maxit %#ok<NASGU>
        g = project_to_feasible_tangent(gradE, v, Q);
        gnorm = sqrt(kahan_sum(abs(g).^2));
        if gnorm <= tol
            return;
        end

        accepted = false;
        alpha = 1.0;
        logf_old = logf;
        v_old = v;

        for ls_it = 1:20 %#ok<NASGU>
            vt = retract_feasible(v + alpha * g, Q);
            if ~isempty(vt)
                [logf_trial, ~, ~, ~] = entropy_streaming_logscore_grad( ...
                    M_gain, A_block, V_old, s2_old, q_old, vt, rows_total, n);
                rhs = logf_old + 1e-4 * alpha * real(g' * g);
                if logf_trial >= rhs
                    accepted = true;
                    v = vt;
                    break;
                end
            end
            alpha = 0.5 * alpha;
        end

        if ~accepted
            v = v_old;
            return;
        end

        [logf, gradE, s2_total, H_approx] = entropy_streaming_logscore_grad( ...
            M_gain, A_block, V_old, s2_old, q_old, v, rows_total, n);

        step_norm = sqrt(kahan_sum(abs(v - v_old).^2));
        f_change = abs(logf - logf_old);
        f_threshold = progress_f_tol * max(1.0, abs(logf_old));

        if f_change <= f_threshold || step_norm <= progress_step_tol
            return;
        end
    end
end

function [logf, g, y2_sq, H] = entropy_logscore_grad_rows(M, v, rows_total, n)
    y = M * v;

    abs_y = abs(y);
    y2_sq = kahan_sum(abs_y.^2);
    y4_4  = kahan_sum(abs_y.^4);

    if y2_sq <= 1e-28 || y4_4 <= 1e-28 || any(~isfinite(y))
        logf = -inf;
        g = zeros(size(v));
        H = inf;
        return;
    end

    c = 2 * log(rows_total / n) / log(rows_total);

    logf = (1 - c) * 0.5 * log(y2_sq) + c * 0.25 * log(y4_4);

    My = M' * y;
    My3 = M' * (y.^3);
    g = (1 - c) * (My / y2_sq) + c * (My3 / y4_4);

    H = -(log(y4_4) - 2 * log(y2_sq));
end

function score = entropy_score_fast(M, v, rows_total, n)
    [logf, ~, ~, ~] = entropy_logscore_grad_rows(M, v, rows_total, n);
    score = exp(logf);
end

function [logf, g, E, Happrox] = entropy_streaming_logscore_grad( ...
    M_gain, A_block, V_old, s2_old, q_old, v, rows_total, n)

    % Gain term from the sketch+new block
    z = M_gain * v;
    gain2 = kahan_sum(abs(z).^2);
    if gain2 <= 1e-28 || any(~isfinite(z))
        logf = -inf;
        g = zeros(size(v));
        E = 0;
        Happrox = inf;
        return;
    end

    % Old-state coefficients
    a = V_old' * v;

    % Exact new-window terms
    y = A_block * v;
    y2_sq = kahan_sum(abs(y).^2);
    y4_4  = kahan_sum(abs(y).^4);

    % Approximate old+new E and Q
    E_old = sum((a.^2) .* s2_old);
    Q_old = sum((a.^4) .* q_old);

    E = E_old + y2_sq;
    Q = Q_old + y4_4;

    if E <= 1e-28 || Q <= 1e-28 || any(~isfinite(y))
        logf = -inf;
        g = zeros(size(v));
        Happrox = inf;
        return;
    end

    gamma = log(rows_total / n) / (2 * log(rows_total));

    % log score = log ||M_gain v||_2 - gamma * Happrox
    %            = 0.5 log(gain2) + gamma log(Q) - 2 gamma log(E)
    logf = 0.5 * log(gain2) + gamma * log(Q) - 2 * gamma * log(E);

    % Gradients
    g_gain = (M_gain' * z) / gain2;

    % dE/dv = 2 * V_old * (s2_old .* a) + 2 * A_block' * y
    gE = 2 * (V_old * (s2_old .* a)) + 2 * (A_block' * y);

    % dQ/dv = 4 * V_old * (q_old .* a.^3) + 4 * A_block' * (y.^3)
    gQ = 4 * (V_old * (q_old .* (a.^3))) + 4 * (A_block' * (y.^3));

    g = g_gain + gamma * (gQ / Q) - 2 * gamma * (gE / E);

    Happrox = -log(Q / (E^2));
end

function g_feas = project_to_feasible_tangent(g, v, Q)
    g_feas = g;
    if ~isempty(Q)
        g_feas = g_feas - Q * (Q' * g_feas);
    end
    g_feas = g_feas - v * (v' * g_feas);
end

function x = project_feasible(x, Q)
    if ~isempty(Q)
        x = x - Q * (Q' * x);
    end
end

function v_new = retract_feasible(x, Q)
    x = project_feasible(x, Q);
    nx = sqrt(kahan_sum(abs(x).^2));
    if nx <= 1e-14
        v_new = [];
    else
        v_new = x / nx;
    end
end

function s = kahan_sum(x)
    x = x(:);
    s = 0;
    c = 0;
    for i = 1:numel(x)
        y = x(i) - c;
        t = s + y;
        c = (t - s) - y;
        s = t;
    end
end

