%% Streaming Forget / Combined / Hybrid experiments over structured CEX variants
% Script-style port for interactive MATLAB debugging.
%
% Modes:
%   "Forget"   - additive forget/window score from cex_structured_new.m.
%   "Combined" - pooled old-row/current-window combined score.
%   "Hybrid"   - first combined_rank directions from Combined, remaining
%                directions from deflated SVD of M_gain.
%   "AugSVD"   - top-r SVD of [Bw; Aw; Beta_w] where Beta_w is a sketch of
%                past rows. beta_mode selects the Beta_w rule:
%                  "reservoir" - reservoir sampling of past rows
%                  "SA"        - cumulative Gaussian RP of past rows
%                  "QQA"       - prototype Q'A using fixed Omega and A_seen
%                              (non-streaming; kept for reference)
%   "iSVD" / "FD" are included as baselines.
%
% This file intentionally leaves diagnostics in the script workspace:
%   diag_history, last_diag, V_r, S_r, state, A, V, svec, old_row_memory,
%   A_block, M_gain, optimizer vectors, oracle vectors, and score components
% are inspectable after the script finishes or after placing a breakpoint.

clear; clc;
if exist('rng', 'file') || exist('rng', 'builtin')
    rng(0);
else
    rand('seed', 0);
    randn('seed', 0);
end
tic

%% Basic parameters
n    = 1024;
r    = 10;
l    = 1;
win  = 32;
mode = "iSVD";      % "Forget", "Combined", "Hybrid", "AugSVD", "iSVD", "FD"
matrix_name = "static-cex"; 
% matrix_name = "mixed-tail-sharp"; 
% matrix_name = "diffuse-diffuse"; 
% matrix_name = sprintf("kernel_stocks_%d_0.2236", n); 
% matrix_name options: 
%   "static-cex" 
%   "diffuse-diffuse"          diffuse signal + diffuse tail
%   "mixed-tail-soft" 
%   "mixed-tail-balanced" 
%   "mixed-tail-sharp" 
%   "risk-residual-panel"      residual/event signal + diffuse heavy-tailed noise
%   "residual-spiky-shocks"    residual/event signal + row-localized spike shocks
%   "spike-diffuse"            alias for "residual-spiky-shocks"
%   "kernel_stocks_<N>_<ls>"   RBF kernel on N synthetic 2D points
%                              (seeded standard-normal stand-in for the stocks
%                              point cloud) with lengthscale ls
%                              (e.g. "kernel_stocks_1000_0.2236"). Overrides n to N.
V_type = "id";        % "id", "U", or "rand"
r_sig = 2;

format compact;
alpha_sig  = 0.003;
alpha_tail = 0.0145;
tail_scale = 0.99;
sigma1     = 1.0;
tail_endpoint = 0.5;    % Optional: set s_n directly for structured spectra, then infer alpha_tail.
shuffle_rows = true;
row_shuffle_seed = 0;

num_restarts = 8;
maxit = 120;
tol = 1e-8;

rownorm_seed_first_block = true;
rownorm_seed_all_blocks = true;
old_memory_size = win;

combined_rank = r;      % Combined: r. Hybrid half at rank 2: set to 1.
if strcmp(mode, "Hybrid")
    combined_rank = floor(r / 2);
end

% AugSVD parameters: only used when mode == "AugSVD".
beta_mode = "QQA";   % "reservoir", "SA", "QQA"
k_beta    = win;           % number of Beta_w rows
% Sketch matrix family used by "SA" and "QQA" beta modes.
%   "gaussian"     - dense i.i.d. standard normal (current default)
%   "sparse_sign"  - sparse_zeta nonzeros per column, each +/-1
%   "count_sketch" - one signed nonzero per column (extreme sparse case)
%   "hadamard"     - SRHT: random sign flips + Hadamard mix + row subsample
%                    (requires the sketched dimension to be supported by
%                    MATLAB's hadamard(), e.g. a power of 2)
sketch_type = "gaussian";
sketch_zeta = 8;           % nonzeros per column for sparse_sign

dump_diagnostics = true;
dump_old_row_responses = false;
dump_blocks = 0;        % 0 means all blocks. Otherwise vector of 1-based block ids.

% Tail-conspiracy plots are most informative for mode="Combined", r=2,
% matrix_name="diffuse-diffuse" or "mixed-tail-sharp".
plot_tail_conspiracy = true;
save_tail_conspiracy_plot = true;
tail_conspiracy_fig_dir = "figures_tail_conspiracy";

% Per-window "kept vs thrown" heatmaps. For each block_id listed, after the
% sketch update we plot two heatmaps of the augmented block matrix M_gain:
%   Kept   = M_gain * V_hat * V_hat'           (lives in carried subspace)
%   Thrown = M_gain - Kept                     (what this window discards)
% Set to [] to disable. Edit freely: e.g. [1 2 6] or 1:r:32 or [1 5 10 20].
kept_vs_thrown_blocks = [1, 2, 6];
save_kept_vs_thrown_plot = true;
kept_vs_thrown_fig_dir = "figures_kept_vs_thrown";

%% --- Matrix setup ---
k = n;
[A, V, svec, sigma1, U] = build_test_matrix( ...
    n, r_sig, alpha_sig, alpha_tail, tail_scale, tail_endpoint, sigma1, V_type, matrix_name);

% Kernel matrices size themselves from the matrix_name; sync n/k to the built A.
if size(A, 1) ~= n
    n = size(A, 1);
    k = n;
    fprintf('Updated n = %d (from %s).\n', n, matrix_name);
end

if ~isempty(tail_endpoint) && ismember(string(matrix_name), ...
        ["static-cex", "diffuse-diffuse", "mixed-tail-soft", "mixed-tail-balanced", "mixed-tail-sharp"])
    alpha_tail = infer_alpha_tail_from_endpoint(n, r_sig, tail_scale, tail_endpoint);
    fprintf('Inferred alpha_tail = %.12g from tail_endpoint = %.12g\n', alpha_tail, tail_endpoint);
end

% Left singular vectors of A BEFORE any row shuffle — shows the natural row
% structure of each direction as the test matrix was constructed.
plot_U_before_stream = true;
if plot_U_before_stream
    [U_exact, ~, ~] = svd(A, "econ");
    figU = figure('Name', sprintf('|U| mesh (pre-shuffle): %s', char(matrix_name)), ...
        'Color', 'w', 'Position', [80, 80, 900, 600]);
    mesh(abs(U_exact));
    xlabel('column (singular vector index)'); ylabel('row'); zlabel('|U_{ij}|');
    title(sprintf('mesh(abs(U)) pre-shuffle: %s, size %dx%d', ...
        char(matrix_name), size(U_exact,1), size(U_exact,2)));
    view(45, 30);
    drawnow;

    figS = figure('Name', sprintf('svec: %s', char(matrix_name)), ...
        'Color', 'w', 'Position', [120, 120, 900, 480]);
    semilogy(1:numel(svec), svec, 'o-', 'LineWidth', 1.2, 'MarkerSize', 4);
    grid on; xlabel('index'); ylabel('singular value (log)');
    title(sprintf('svec: %s  (n=%d, \\sigma_1=%.4g)', char(matrix_name), numel(svec), svec(1)));
    drawnow;
end

if shuffle_rows
    if exist('rng', 'file') || exist('rng', 'builtin')
        rng(row_shuffle_seed);
    else
        rand('seed', row_shuffle_seed);
        randn('seed', row_shuffle_seed);
    end
    p = randperm(n);
    A = A(p, :);
end

if r < k
    E_opt = sum(svec(r+1:end).^2);
else
    E_opt = 0;
end

%% Streaming state
state = [];
V_r = [];
S_r = [];
H_r = [];
score_r = [];
A_seen = zeros(0, n);
diag_history = struct([]);
tail_diag_history = struct([]);
prev_selected_frame = [];
prev_carried_frame = [];
oracle_V_r = [];
oracle_S_r = [];

% AugSVD per-variant state (only meaningful for mode == "AugSVD").
beta_state = struct();
beta_state.buf = zeros(0, n);             % reservoir buffer
beta_state.rows_seen_total = 0;           % reservoir counter
beta_state.SA = zeros(0, n);              % cumulative Gaussian sketch
Omega_qqa = [];
if strcmp(mode, "AugSVD")
    if strcmp(beta_mode, "SA")
        beta_state.SA = zeros(k_beta, n);
    elseif strcmp(beta_mode, "QQA")
        Omega_qqa = build_sketch_matrix(k_beta, n, sketch_type, sketch_zeta);
    end
end

% disp(combined_components([1 1; 0 0], [0 0], [0 0], 0, 2, [0;1]));
% theta = linspace(0, 2*pi, 2000);
% v = [cos(theta); sin(theta)];   % 2 x N
% Z = zeros(size(theta));
% for i = 1:length(theta)
%     vi = v(:, i);
%     out = forget_components_initial([0.5 1; 0.5 0], vi, 100);
%     Z(i) = out.total;
% end
% figure;
% plot3(v(1,:), v(2,:), Z, 'LineWidth', 2);
% xlabel('v_1'); ylabel('v_2'); zlabel('score');
% grid on;


%% Streaming over row blocks
mA = size(A, 1);
block_id = 0;
for start_row = 1:win:mA
    block_id = block_id + 1;

    % if block_id == 3
    %     break
    % end

    end_row = min(start_row + win - 1, mA);
    A_block = A(start_row:end_row, :);

    if isempty(V_r) || isempty(S_r)
        prev_sketch = [];
        M_gain = A_block;
    else
        prev_sketch = S_r * V_r';
        M_gain = [prev_sketch; A_block];
    end

    old_row_memory = select_old_row_memory(A_seen, old_memory_size);
    rows_seen = end_row;

    should_dump = dump_diagnostics && (isequal(dump_blocks, 0) || any(dump_blocks == block_id));

    fd_info = [];   % populated only by the iSVD/FD branch; consumed by kept-vs-thrown plot

    switch mode
        case "Forget"
            if isempty(state)
                fprintf('\n===== block rows %d:%d (initial forget/window score) =====\n', start_row, end_row);
            else
                fprintf('\n===== block rows %d:%d (streaming forget/window score) =====\n', start_row, end_row);
            end

            V_seed = rownorm_seed_if_enabled(A_block, r, block_id, rownorm_seed_first_block, rownorm_seed_all_blocks);
            [V_hat, score_new, H_new, state_new, diag_block] = ...
                iter_basis_forget(A_block, r, n, state, V_seed, num_restarts, maxit, tol, should_dump, M_gain, V);

        case {"Combined", "Hybrid"}
            if isempty(state)
                fprintf('\n===== block rows %d:%d (initial combined score) =====\n', start_row, end_row);
            else
                fprintf('\n===== block rows %d:%d (streaming combined score) =====\n', start_row, end_row);
            end

            V_seed = rownorm_seed_if_enabled(A_block, r, block_id, rownorm_seed_first_block, rownorm_seed_all_blocks);
            cr = min(max(combined_rank, 0), r);
            [V_hat, score_new, H_new, state_new, diag_block] = ...
                iter_basis_combined_hybrid(A_block, M_gain, old_row_memory, rows_seen, n, r, cr, ...
                    V_seed, num_restarts, maxit, tol, should_dump, V);

        case "AugSVD"
            if isempty(state)
                fprintf('\n===== block rows %d:%d (initial AugSVD, beta_mode=%s) =====\n', ...
                    start_row, end_row, beta_mode);
            else
                fprintf('\n===== block rows %d:%d (streaming AugSVD, beta_mode=%s) =====\n', ...
                    start_row, end_row, beta_mode);
            end

            Beta_w = augsvd_beta_compute(beta_mode, beta_state, A_seen, Omega_qqa, n);
            if isempty(Beta_w)
                M_aug = M_gain;
            else
                % M_aug = [M_gain; Beta_w];
                M_aug = [Beta_w];
            end
            [~, S_aug, V_aug] = svd(M_aug, "econ");
            s_aug = diag(S_aug);
            rr = min(r, numel(s_aug));
            V_hat = V_aug(:, 1:rr);
            s_new = s_aug(1:rr);
            score_new = s_new.^2;
            H_new = nan(rr, 1);
            state_new = struct();
            state_new.V = V_hat;
            state_new.s = s_new;
            state_new.s2 = s_new.^2;
            state_new.rows_seen = rows_seen;
            diag_block = struct();
            diag_block.block_id = block_id;
            diag_block.rows = [start_row, end_row];

        case {"iSVD", "FD"}
            if isempty(V_r)
                M = A_block;
            else
                M = [S_r * V_r'; A_block];
            end
            [~, S_hat, V_hat_full] = svd(M, "econ");
            s = diag(S_hat);
            rr = min(r, numel(s));
            switch mode
                case "iSVD"
                    V_hat = V_hat_full(:, 1:rr);
                    s_new = s(1:rr);
                    delta = 0;
                case "FD"
                    if numel(s) > rr
                        delta = s(rr+1)^2;
                    else
                        delta = 0;
                    end
                    s_new = sqrt(max(s(1:rr).^2 - delta, 0));
                    V_hat = V_hat_full(:, 1:rr);
            end
            fd_info = struct();
            fd_info.mode  = mode;
            fd_info.s_top = s(1:rr);   % top-r sigvals of augmented M_gain
            fd_info.s_new = s_new;     % what FD/iSVD actually carries forward
            fd_info.delta = delta;     % FD shrinkage scalar (0 for iSVD)
            score_new = s_new.^2;
            H_new = nan(size(s_new));
            state_new = struct();
            state_new.V = V_hat;
            state_new.s = s_new;
            state_new.s2 = s_new.^2;
            state_new.rows_seen = rows_seen;
            diag_block = struct();
            diag_block.block_id = block_id;
            diag_block.rows = [start_row, end_row];

        otherwise
            error('Unknown mode.');
    end

    V_selected = V_hat;
    score_selected = score_new; %#ok<NASGU>
    H_selected = H_new; %#ok<NASGU>

    % Compress the carried sketch to the selected rank-r subspace for
    % Forget/Combined/Hybrid/AugSVD, matching cex_structured_new.m style.
    % AugSVD projects onto M_gain (Beta_w influences direction selection
    % only, not the carried singular values).
    if strcmp(mode, "Forget") || strcmp(mode, "Combined") || strcmp(mode, "Hybrid") || strcmp(mode, "AugSVD")
        [V_hat, s_new] = projected_subspace_svd(M_gain, V_hat);
        state_new.V = V_hat;
        state_new.s = s_new;
        state_new.s2 = s_new.^2;
        state_new.rows_seen = rows_seen;
        if strcmp(mode, "AugSVD")
            % AugSVD has no separate score function — keep score_new paired
            % with the post-projection singular values, like iSVD does.
            score_new = s_new.^2;
        end
    end

    V_carried = V_hat;

    if ~isempty(kept_vs_thrown_blocks) && any(kept_vs_thrown_blocks == block_id)
        plot_kept_vs_thrown_decomposition( ...
            M_gain, V_hat, block_id, [start_row, end_row], matrix_name, mode, r, ...
            save_kept_vs_thrown_plot, kept_vs_thrown_fig_dir, fd_info);
    end

    if plot_tail_conspiracy && r >= 2 && (strcmp(mode, "Combined") || strcmp(mode, "Hybrid"))
        if isempty(oracle_V_r) || isempty(oracle_S_r)
            M_oracle_gain = A_block;
        else
            M_oracle_gain = [oracle_S_r * oracle_V_r'; A_block];
        end
        M_zero_gain = A_block;
        tail_diag = build_tail_conspiracy_diag( ...
            block_id, [start_row, end_row], A_block, M_gain, M_zero_gain, M_oracle_gain, ...
            old_row_memory, rows_seen, n, V_selected, V_carried, V, ...
            prev_selected_frame, prev_carried_frame);
        tail_diag_history = [tail_diag_history; tail_diag]; %#ok<AGROW>

        Q_oracle_forced = projected_oracle_frame(M_oracle_gain, V, r);
        [oracle_V_r, oracle_s_new] = projected_subspace_svd(M_oracle_gain, Q_oracle_forced);
        oracle_S_r = diag(oracle_s_new);
    end

    V_r = V_hat;
    S_r = diag(s_new);
    H_r = H_new;
    score_r = score_new;
    state = state_new;

    A_seen = [A_seen; A_block]; %#ok<AGROW>

    if strcmp(mode, "AugSVD")
        beta_state = augsvd_beta_update(beta_mode, beta_state, A_block, k_beta, sketch_type, sketch_zeta);
    end

    fprintf('rows %d:%d\n', start_row, end_row);
    fprintf('s: '); disp(s_new(:)');
    fprintf('H: '); disp(H_new(:)');
    fprintf('scores: '); disp(score_new(:)');

    diag_block.block_id = block_id;
    diag_block.rows = [start_row, end_row];
    diag_block.A_block = A_block;
    diag_block.M_gain = M_gain;
    diag_block.old_row_memory = old_row_memory;
    diag_block.V_score = V_hat;
    diag_block.s = s_new;
    diag_block.score = score_new;
    diag_history = [diag_history; diag_block]; %#ok<AGROW>
    last_diag = diag_block; %#ok<NASGU>

    if should_dump && (strcmp(mode, "Forget") || strcmp(mode, "Combined") || strcmp(mode, "Hybrid"))
        print_projection_summary(V_hat, V, M_gain);
        if dump_old_row_responses && isfield(diag_block, 'oracle_raw') && ~isempty(old_row_memory)
            fprintf('old_row_memory @ oracle_raw vectors:\n');
            for jj = 1:numel(diag_block.oracle_raw)
                disp((old_row_memory * diag_block.oracle_raw(jj).v).');
            end
        end
    end

    prev_selected_frame = V_selected;
    prev_carried_frame = V_carried;
end

if plot_tail_conspiracy && ~isempty(tail_diag_history)
    plot_tail_conspiracy_results(tail_diag_history, matrix_name, mode, save_tail_conspiracy_plot, tail_conspiracy_fig_dir);
end

%% Metrics after full pass
align = norm(V_r * V_r' * V(:,1), 'fro');

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
Delta = E_alg - E_opt; %#ok<NASGU>

summary_table = table(string(matrix_name), string(mode), sigma1, align, rel_err_sval, ...
    'VariableNames', {'matrix', 'mode', 'sigma1', 'mean_align', 'mean_relerr_sval'});
disp(summary_table);

elapsedTime = toc;
fprintf('Elapsed time: %.3f\n', elapsedTime);

%% ========================= Local helper functions =========================

function [A, V_exact, svec, sigma1, U] = build_test_matrix( ...
    n, r_sig, alpha_sig, alpha_tail, tail_scale, tail_endpoint, sigma1, V_type, matrix_name)

    if startsWith(string(matrix_name), "kernel_")
        [A, V_exact, svec, sigma1] = build_kernel_matrix(matrix_name);
        U = V_exact;
        return;
    end

    switch matrix_name
        case {"static-cex", "diffuse-diffuse", "mixed-tail-soft", "mixed-tail-balanced", "mixed-tail-sharp"}
            [U, V_exact, svec] = build_structured_matrix_factors( ...
                n, r_sig, alpha_sig, alpha_tail, tail_scale, tail_endpoint, sigma1, V_type, matrix_name);
            A = U * diag(svec) * V_exact';

        case "risk-residual-panel"
            [A, V_exact, svec, sigma1, U] = build_risk_residual_panel(n, "fast");

        case {"residual-spiky-shocks", "spike-diffuse"}
            [A, V_exact, svec, sigma1, U] = build_residual_spiky_shocks(n, "fast");

        otherwise
            error('Unknown matrix_name.');
    end
end

function [U, V, svec] = build_structured_matrix_factors( ...
    n, r_sig, alpha_sig, alpha_tail, tail_scale, tail_endpoint, sigma1, V_type, matrix_name)

    H = hadamard(n);
    U_sig = H(:, 1:r_sig) / sqrt(n);

    switch matrix_name
        case "static-cex"
            U0 = zeros(n, n);
            U0(:, 1:r_sig) = U_sig;
            a_tail = sqrt(1 - r_sig/n);
            b_tail = 1/sqrt(n);
            for j = r_sig+1:n
                col = zeros(n,1);
                idx_large = j - r_sig;
                col(idx_large) = a_tail;
                col(n-r_sig+1:n) = b_tail;
                U0(:, j) = col;
            end
            [U, ~] = qr(U0, 0);

        case "diffuse-diffuse"
            G = randn(n, n-r_sig);
            G = G - U_sig * (U_sig' * G);
            [U_tail, ~] = qr(G, 0);
            [U, ~] = qr([U_sig, U_tail], 0);

        case {"mixed-tail-soft", "mixed-tail-balanced", "mixed-tail-sharp"}
            switch matrix_name
                case "mixed-tail-soft"
                    tail_spikiness = 0.25;
                case "mixed-tail-balanced"
                    tail_spikiness = 0.50;
                case "mixed-tail-sharp"
                    tail_spikiness = 0.75;
            end
            G = randn(n, n-r_sig);
            G = G - U_sig * (U_sig' * G);
            [U_diffuse_tail, ~] = qr(G, 0);

            U_spiky_raw = zeros(n, n-r_sig);
            a_tail = sqrt(1 - r_sig/n);
            b_tail = 1/sqrt(n);
            for j = 1:n-r_sig
                col = zeros(n,1);
                col(j) = a_tail;
                col(n-r_sig+1:n) = b_tail;
                U_spiky_raw(:, j) = col;
            end
            U_spiky_raw = U_spiky_raw - U_sig * (U_sig' * U_spiky_raw);
            [U_spiky_tail, ~] = qr(U_spiky_raw, 0);

            tail_raw = sqrt(1-tail_spikiness) * U_diffuse_tail + sqrt(tail_spikiness) * U_spiky_tail;
            tail_raw = tail_raw - U_sig * (U_sig' * tail_raw);
            [U_tail, ~] = qr(tail_raw, 0);
            [U, ~] = qr([U_sig, U_tail], 0);

        otherwise
            error('Unknown matrix_name.');
    end

    for j = 1:r_sig
        if dot(U(:, j), U_sig(:, j)) < 0
            U(:, j) = -U(:, j);
        end
    end

    switch V_type
        case "id"
            V = eye(n);
        case "U"
            V = U;
        case "rand"
            [V, ~] = qr(randn(n, n), 0);
        otherwise
            error('Unknown V_type.');
    end

    if ~isempty(tail_endpoint)
        alpha_tail = infer_alpha_tail_from_endpoint(n, r_sig, tail_scale, tail_endpoint);
    end

    sig_block  = sigma1 * (1:r_sig).^(-alpha_sig);
    tail_block = tail_scale * (1:(n-r_sig)).^(-alpha_tail);

    % sig_block  = logspace(log10(tail_scale),0,r_sig+1);
    % sig_block  = flip(sig_block(1:r_sig));
    % tail_block = flip(logspace(log10(tail_scale * (n-r_sig)^(-alpha_tail)),log10(tail_scale),n-r_sig));
    svec = [sig_block, tail_block];
    svec(1) = sigma1;
end

function alpha_tail = infer_alpha_tail_from_endpoint(n, r_sig, tail_scale, tail_endpoint)
    tail_len = n - r_sig;
    if tail_len < 1
        error('Cannot infer alpha_tail when n - r_sig < 1.');
    end
    if tail_endpoint <= 0
        error('tail_endpoint must be positive.');
    end
    if tail_scale <= 0
        error('tail_scale must be positive.');
    end
    if tail_len == 1
        alpha_tail = 0;
        return;
    end
    alpha_tail = -log(tail_endpoint / tail_scale) / log(tail_len);
end

function [A, V_exact, svec, sigma1, U] = build_risk_residual_panel(n, preset)
    if n <= 8
        error('n must be larger than 8 for risk-residual-panel.');
    end
    t = linspace(0, 1, n+1)';
    t = t(1:n);
    groups = min(8, max(4, floor(n / 16)));
    asset_groups = split_indices(n, groups);

    V0 = zeros(n, n);
    crowding = zeros(n, 1);
    liquidity = zeros(n, 1);
    for g = 1:groups
        idx = asset_groups{g};
        local = linspace(-1, 1, numel(idx))';
        side = 1;
        if mod(g-1, 2) == 1
            side = -1;
        end
        crowding(idx) = side * (1 + 0.25 * sin(pi * local));
        liquidity(idx) = (local - mean(local)) * (1 + 0.12 * (g-1));
    end
    V0(:,1) = crowding - mean(crowding);
    V0(:,2) = liquidity - mean(liquidity);
    for j = 3:n
        idx0 = j - 3;
        group = asset_groups{mod(idx0, groups) + 1};
        V0(idx0 + 1, j) = 1;
        V0(:, j) = V0(:, j) + 0.015 * randn(n,1);
        V0(group, j) = V0(group, j) + 0.08 * randn(numel(group),1);
        V0(:, j) = V0(:, j) - mean(V0(:, j));
    end
    [Vbase, ~] = qr(V0, 0);
    for j = 1:2
        if dot(Vbase(:,j), V0(:,j)) < 0
            Vbase(:,j) = -Vbase(:,j);
        end
    end

    [Ubase, s_design] = residual_panel_left_factors(n, preset);
    A = (Ubase .* s_design) * Vbase';
    vol = 0.82 + 0.5 * exp(-0.5 * ((t - 0.66) / 0.12).^2);
    vol = vol + 0.08 * sin(2 * pi * 13 * t);
    A = vol .* A;
    if strcmp(preset, "small")
        idio_scale = 0.006;
    else
        idio_scale = 0.007;
    end
    A = A + idio_scale * student_t_randn(5, size(A)) / sqrt(5 / 3);
    A = A - mean(A, 1);

    [U, S_exact, Vh] = svd(A, "econ");
    svec = diag(S_exact)';
    V_exact = Vh;
    sigma1 = svec(1);
end

function [A, V_exact, svec, sigma1, U] = build_residual_spiky_shocks(n, preset)
    if n <= 8
        error('n must be larger than 8 for residual-spiky-shocks.');
    end
    groups = min(8, max(4, floor(n / 16)));
    asset_groups = split_indices(n, groups);

    V0 = zeros(n, n);
    crowding = zeros(n, 1);
    liquidity = zeros(n, 1);
    for g = 1:groups
        idx = asset_groups{g};
        local = linspace(-1, 1, numel(idx))';
        side = 1;
        if mod(g-1, 2) == 1
            side = -1;
        end
        crowding(idx) = side * (1 + 0.25 * sin(pi * local));
        liquidity(idx) = (local - mean(local)) * (1 + 0.12 * (g-1));
    end
    V0(:,1) = crowding - mean(crowding);
    V0(:,2) = liquidity - mean(liquidity);
    for j = 3:n
        idx0 = j - 3;
        group = asset_groups{mod(idx0, groups) + 1};
        V0(idx0 + 1, j) = 1;
        V0(:, j) = V0(:, j) + 0.015 * randn(n,1);
        V0(group, j) = V0(group, j) + 0.08 * randn(numel(group),1);
        V0(:, j) = V0(:, j) - mean(V0(:, j));
    end
    [Vbase, ~] = qr(V0, 0);
    for j = 1:2
        if dot(Vbase(:,j), V0(:,j)) < 0
            Vbase(:,j) = -Vbase(:,j);
        end
    end

    [Ubase, s_design] = residual_panel_left_factors(n, preset);
    A = (Ubase .* s_design) * Vbase';

    if strcmp(preset, "small")
        spike_rate = 0.015;
        spike_amp = 0.10;
        noise_floor = 0.001;
    else
        spike_rate = 0.02;
        spike_amp = 0.10;
        noise_floor = 0.0012;
    end

    n_spike_rows = max(2, round(spike_rate * n));
    spike_rows = randperm(n, n_spike_rows);
    support_size = max(2, floor(n / 64));
    spike_support = randperm(n, support_size);
    spike_direction = randn(support_size, 1);
    spike_direction = spike_direction / max(norm(spike_direction), 1e-30);
    for ii = 1:numel(spike_rows)
        A(spike_rows(ii), spike_support) = A(spike_rows(ii), spike_support) + spike_amp * spike_direction';
    end

    A = A + noise_floor * randn(n, n);
    A = A - mean(A, 1);

    [U, S_exact, Vh] = svd(A, "econ");
    svec = diag(S_exact)';
    V_exact = Vh;
    sigma1 = svec(1);
end

function [A, V_exact, svec, sigma1] = build_kernel_matrix(matrix_name)
    % Mirrors the "kernel_<shape>_<num_points>_<lengthscale>" branch in main.py
    % but self-contained: builds the point cloud synthetically (no file I/O),
    % then forms the n×n RBF kernel A_ij = exp(-||x_i - x_j||^2 / (2 ls^2))
    % and returns its exact right singular system.
    parts = split(string(matrix_name), "_");
    if numel(parts) < 4
        error('Kernel matrix_name must be "kernel_<shape>_<num_points>_<lengthscale>", got "%s".', matrix_name);
    end
    shape_name = parts(end-2);
    num_points = str2double(parts(end-1));
    lengthscale = str2double(parts(end));
    if ~isfinite(num_points) || num_points <= 0 || ~isfinite(lengthscale) || lengthscale <= 0
        error('Could not parse num_points / lengthscale from "%s".', matrix_name);
    end
    num_points = round(num_points);

    if exist('rng', 'file') || exist('rng', 'builtin')
        rng(10);
    else
        rand('seed', 10);
        randn('seed', 10);
    end

    switch shape_name
        case "stocks"
            % Self-contained stand-in for ~/data/data_2m.mtx: deterministic 2D
            % standard-normal points. Same dimensionality (n×2) and roughly the
            % same scale as the original log-returns file.
            points = randn(num_points, 2);
        otherwise
            error('Kernel shape "%s" not supported.', shape_name);
    end

    fprintf('Building RBF kernel: shape=%s, num_points=%d, lengthscale=%g.\n', ...
        shape_name, num_points, lengthscale);
    A = rbf_kernel_matrix(points, lengthscale);
    [~, S_exact, Vh] = svd(A, "econ");
    svec = diag(S_exact)';
    V_exact = Vh;
    sigma1 = svec(1);
    fprintf('Kernel matrix built: size %dx%d, sigma1 = %.6g.\n', size(A,1), size(A,2), sigma1);
end

function K = rbf_kernel_matrix(points, lengthscale)
    sq = sum(points .* points, 2);
    D2 = sq + sq' - 2 * (points * points');
    D2 = max(D2, 0);
    K = exp(-D2 / (2 * lengthscale^2));
end

function [Ubase, s_design] = residual_panel_left_factors(n, preset)
    t = linspace(0, 1, n+1)';
    t = t(1:n);
    U0 = zeros(n, n);
    crowding_event = exp(-0.5 * ((t - 0.64) / 0.105).^2);
    crowding_event = crowding_event + 0.42 * exp(-0.5 * ((t - 0.22) / 0.055).^2);
    crowding_event = crowding_event .* (1 + 0.2 * sin(2 * pi * 5 * t));
    residual_turnover = sin(2 * pi * 3 * t + 0.35);
    residual_turnover = residual_turnover + 0.45 * sin(2 * pi * 11 * t);
    residual_turnover = residual_turnover .* (0.8 + 0.5 * exp(-0.5 * ((t - 0.78) / 0.09).^2));
    U0(:,1) = crowding_event - mean(crowding_event);
    U0(:,2) = residual_turnover - mean(residual_turnover);
    for j = 3:n
        center = (j - 2.5) / max(1, n - 2);
        width = 0.012 + 0.018 * (mod(j-1, 5) / 4.0);
        wrapped = mod(t - center + 0.5, 1.0) - 0.5;
        pulse = exp(-0.5 * (wrapped / width).^2);
        pulse = pulse + 0.03 * randn(n,1);
        U0(:,j) = pulse - mean(pulse);
    end
    [Ubase, ~] = qr(U0, 0);
    for j = 1:2
        if dot(Ubase(:,j), U0(:,j)) < 0
            Ubase(:,j) = -Ubase(:,j);
        end
    end
    if strcmp(preset, "small")
        signal_gap = 0.012;
        tail_scale_local = 0.975;
    else
        signal_gap = 0.010;
        tail_scale_local = 0.982;
    end
    s_signal = [1.0, 1.0 - signal_gap];
    tail_len = n - numel(s_signal);
    tail = tail_scale_local * (1:tail_len).^(-0.035);
    tail = tail .* (1 + 0.018 * sin((0:tail_len-1) * 0.71));
    s_design = [s_signal, tail];
end

function groups = split_indices(n, num_groups)
    edges = round(linspace(0, n, num_groups + 1));
    groups = cell(num_groups, 1);
    for g = 1:num_groups
        groups{g} = (edges(g)+1):edges(g+1);
    end
end

function X = student_t_randn(df, sz)
    Z = randn(sz);
    chi = zeros(sz);
    for jj = 1:df
        chi = chi + randn(sz).^2;
    end
    X = Z ./ sqrt(chi / df);
end

function old_rows = select_old_row_memory(A_seen, old_memory_size)
    if isempty(A_seen) || old_memory_size <= 0
        old_rows = zeros(0, size(A_seen, 2));
        return;
    end
    take = min(old_memory_size, size(A_seen, 1));
    old_rows = A_seen(end-take+1:end, :);
end

function V_seed = rownorm_seed_if_enabled(A_block, r, block_id, first_enabled, all_enabled)
    if (block_id == 1 && first_enabled) || (block_id > 1 && all_enabled)
        row_norms = sqrt(sum(A_block.^2, 2));
        row_norms(row_norms < 1e-30) = 1;
        A_norm = A_block ./ row_norms;
        [~, ~, Vsvd] = svd(A_norm, "econ");
        V_seed = Vsvd(:, 1:min(r, size(Vsvd,2)));
    else
        V_seed = [];
    end
end

function [V_out, score_out, H_out, state_out, diag_block] = iter_basis_forget( ...
    A_block, r, n, state_prev, V_seed, num_restarts, maxit, tol, do_diag, M_gain, V_exact)

    d = size(A_block, 2);
    V_out = zeros(d, r);
    score_out = -inf(r, 1);
    H_out = nan(r, 1);
    Q = zeros(d, 0);
    is_initial = isempty(state_prev);
    if is_initial
        prev_basis = [];
        prev_s2 = [];
    else
        prev_basis = state_prev.V;
        prev_s2 = state_prev.s2;
    end
    diag_block = struct();

    for kk = 1:r
        starts = make_basic_restart_seeds(M_gain, Q, kk, V_seed, num_restarts);
        best_v = [];
        best_score = -inf;
        best_H = nan;
        for rr = 1:num_restarts
            v0 = starts{rr};
            if is_initial
                [v_cand, score_cand, H_cand] = basic_projected_ascent(v0, Q, maxit, tol, ...
                    @(v) forget_score_grad_initial(A_block, v, n));
            else
                [v_cand, score_cand, H_cand] = basic_projected_ascent(v0, Q, maxit, tol, ...
                    @(v) forget_score_grad_streaming(A_block, prev_basis, prev_s2, v, n));
            end
            if score_cand > best_score
                best_v = v_cand;
                best_score = score_cand;
                best_H = H_cand;
            end
        end
        Q = [Q, best_v]; %#ok<AGROW>
        V_out(:, kk) = best_v;
        score_out(kk) = best_score;
        H_out(kk) = best_H;
    end

    state_out = struct();
    state_out.V = V_out;
    state_out.s = sqrt(max(sum((M_gain * V_out).^2, 1), 0)).';
    state_out.s2 = state_out.s.^2;
    state_out.rows_seen = size(M_gain,1);

    if do_diag
        diag_block = build_forget_diag(A_block, M_gain, V_out, V_exact, n, prev_basis, prev_s2, is_initial);
        print_forget_diag(diag_block);
    end
end

function [V_out, score_out, H_out, state_out, diag_block] = iter_basis_combined_hybrid( ...
    A_block, M_gain, old_row_memory, rows_seen, n, r, combined_rank, V_seed, ...
    num_restarts, maxit, tol, do_diag, V_exact)

    d = size(A_block, 2);
    V_out = zeros(d, r);
    score_out = -inf(r, 1);
    H_out = nan(r, 1);
    Q = zeros(d, 0);
    diag_block = struct();

    for kk = 1:combined_rank
        starts = make_basic_restart_seeds(M_gain, Q, kk, V_seed, num_restarts);
        best_v = [];
        best_score = -inf;
        best_H = nan;
        for rr = 1:num_restarts
            v0 = starts{rr};
            [v_cand, score_cand, H_cand] = basic_projected_ascent(v0, Q, maxit, tol, ...
                @(v) combined_score_grad(A_block, M_gain, old_row_memory, rows_seen, n, v));
            if score_cand > best_score
                best_v = v_cand;
                best_score = score_cand;
                best_H = H_cand;
            end
        end
        Q = [Q, best_v]; %#ok<AGROW>
        V_out(:, kk) = best_v;
        score_out(kk) = best_score;
        H_out(kk) = best_H;
    end

    if combined_rank < r
        M_def = M_gain * (eye(d) - Q*Q');
        [~, ~, Vsvd] = svd(M_def, "econ");
        for kk = combined_rank+1:r
            v = retract_feasible(Vsvd(:, kk-combined_rank), Q);
            Q = [Q, v]; %#ok<AGROW>
            V_out(:, kk) = v;
            [score_out(kk), ~, ~, H_out(kk)] = combined_score_grad(A_block, M_gain, old_row_memory, rows_seen, n, v);
        end
    end

    state_out = struct();
    state_out.V = V_out;
    state_out.s = sqrt(max(sum((M_gain * V_out).^2, 1), 0)).';
    state_out.s2 = state_out.s.^2;
    state_out.rows_seen = rows_seen;

    if do_diag
        diag_block = build_combined_diag(A_block, M_gain, old_row_memory, rows_seen, n, V_out, V_exact);
        print_combined_diag(diag_block);
    end
end

function [v, score, H] = basic_projected_ascent(v0, Q, maxit, tol, score_grad_fun)
    v = retract_feasible(v0, Q);
    if isempty(v)
        v = retract_feasible(randn(size(Q, 1), 1), Q);
    end
    if isempty(v)
        error('Could not construct a feasible initial vector.');
    end
    [score, grad, ~, H] = score_grad_fun(v);
    for it = 1:maxit
        g = project_to_feasible_tangent(grad, v, Q);
        ng = norm(g);
        if ng < tol
            break;
        end
        score_old = score;
        alpha = 1.0;
        accepted = false;
        for ls = 1:25
            vt = retract_feasible(v + alpha * g, Q);
            if isempty(vt)
                alpha = 0.5 * alpha;
                continue;
            end
            [score_trial, ~, ~, ~] = score_grad_fun(vt);
            if score_trial >= score_old + 1e-4 * alpha * real(g' * g)
                v = vt;
                score = score_trial;
                accepted = true;
                break;
            end
            alpha = 0.5 * alpha;
        end
        [score, grad, ~, H] = score_grad_fun(v);
        if ~accepted || abs(score - score_old) < 1e-10 * max(1, abs(score_old))
            break;
        end
    end
end

function starts = make_basic_restart_seeds(M, Q, k, V_seed, num_restarts)
    d = size(M, 2);
    starts = cell(num_restarts, 1);
    [~, ~, Vsvd] = svd(M, "econ");
    num_top = min(4, size(Vsvd, 2));
    for restart = 1:num_restarts
        restart_type = mod(restart - 1, 5) + 1;
        block = floor((restart - 1) / 5);
        switch restart_type
            case 1
                if ~isempty(V_seed) && size(V_seed,2) >= k
                    v0 = V_seed(:, k);
                else
                    v0 = Vsvd(:, 1);
                end
            case 2
                v0 = Vsvd(:, mod(block, num_top) + 1);
            case 3
                j1 = mod(block, num_top) + 1;
                j2 = mod(block + 1, num_top) + 1;
                v0 = 0.75 * Vsvd(:, j1) + sqrt(1 - 0.75^2) * Vsvd(:, j2);
            case 4
                v0 = Vsvd(:, mod(block, num_top) + 1) + 1e-2 * randn(d,1);
            otherwise
                v0 = randn(d,1);
        end
        v = retract_feasible(v0, Q);
        if isempty(v)
            for attempt = 1:20
                v = retract_feasible(randn(d,1), Q);
                if ~isempty(v)
                    break;
                end
            end
        end
        if isempty(v)
            error('Could not generate feasible restart seed.');
        end
        starts{restart} = v;
    end
end

function [score, g, y2_sq, H] = forget_score_grad_initial(M, v, n)
    y = M * v;
    y2_sq = kahan_sum(y.^2);
    y4_4 = kahan_sum(y.^4);
    rows_new = size(M, 1);
    if rows_new <= 1 || y2_sq <= 1e-28 || y4_4 <= 1e-28
        score = 0; g = zeros(size(v)); H = inf; return;
    end
    c_k = log(rows_new / n) / log(rows_new);
    scale = (rows_new / n)^(1/4);
    score = scale * exp((1 - 0.5 * c_k) * log(y2_sq) + 0.25 * c_k * log(y4_4));
    g = score * ((2 - c_k) * (M' * y / y2_sq) + c_k * (M' * (y.^3) / y4_4));
    H = -(log(y4_4) - 2 * log(y2_sq));
end

function [score, g, P_total, Hcurr] = forget_score_grad_streaming(A_block, V_old, s2_old, v, n)
    a = V_old' * v;
    y = A_block * v;
    y2_sq = kahan_sum(y.^2);
    y4_4 = kahan_sum(y.^4);
    P_old = sum((a.^2) .* s2_old);
    g_old = 2 * (V_old * (s2_old .* a));
    rows_new = size(A_block, 1);
    if rows_new <= 1 || y2_sq <= 1e-28 || y4_4 <= 1e-28
        score = P_old; g = g_old; P_total = P_old; Hcurr = inf; return;
    end
    c_k = log(rows_new / n) / log(rows_new);
    scale = (rows_new / n)^(1/4);
    psi = scale * exp((1 - 0.5 * c_k) * log(y2_sq) + 0.25 * c_k * log(y4_4));
    g_psi = psi * ((2 - c_k) * (A_block' * y / y2_sq) + c_k * (A_block' * (y.^3) / y4_4));
    score = P_old + psi;
    g = g_old + g_psi;
    P_total = P_old + y2_sq;
    Hcurr = -(log(y4_4) - 2 * log(y2_sq));
end

function [score, g, gain2, H] = combined_score_grad(A_block, M_gain, old_row_memory, rows_seen, n, v)
    y_gain = M_gain * v;
    gain2 = kahan_sum(y_gain.^2);
    R = [old_row_memory; A_block];
    z = R * v;
    z2 = kahan_sum(z.^2);
    z4 = kahan_sum(z.^4);
    rows_entropy = size(R, 1);
    if rows_entropy <= 1 || gain2 <= 1e-28 || z2 <= 1e-28 || z4 <= 1e-28
        score = gain2; g = 2 * (M_gain' * y_gain); H = inf; return;
    end
    c = log(rows_seen / n) / (2 * log(rows_entropy));
    log_ratio = log(z4) - 2 * log(z2);
    phi = exp(c * log_ratio);
    score = gain2 * phi;
    H = -log_ratio;
    grad_gain = 2 * (M_gain' * y_gain);
    grad_log_ratio = 4 * (R' * (z.^3) / z4) - 4 * (R' * z / z2);
    g = phi * grad_gain + gain2 * phi * c * grad_log_ratio;
end

function diag_block = build_combined_diag(A_block, M_gain, old_row_memory, rows_seen, n, V_out, V_exact)
    diag_block = struct();
    diag_block.oracle_raw = struct([]);
    diag_block.optimizer = struct([]);
    Q_oracle_cols = zeros(size(V_out,1), size(V_out,2));
    for jj = 1:size(V_out,2)
        vraw = M_gain' * (M_gain * V_exact(:, jj));
        if norm(vraw) > 1e-14
            vraw = vraw / norm(vraw);
        end
        Q_oracle_cols(:, jj) = vraw;
        diag_block.oracle_raw(jj).v = vraw;
        diag_block.oracle_raw(jj).comp = combined_components(A_block, M_gain, old_row_memory, rows_seen, n, vraw);
        diag_block.optimizer(jj).v = V_out(:, jj);
        diag_block.optimizer(jj).comp = combined_components(A_block, M_gain, old_row_memory, rows_seen, n, V_out(:, jj));
    end
    [Q_oracle, ~] = qr(Q_oracle_cols, 0);
    diag_block.Q_oracle = Q_oracle;
    diag_block.oracle_qr_sum = 0;
    for jj = 1:size(Q_oracle,2)
        comp = combined_components(A_block, M_gain, old_row_memory, rows_seen, n, Q_oracle(:, jj));
        diag_block.oracle_qr(jj).v = Q_oracle(:, jj); %#ok<AGROW>
        diag_block.oracle_qr(jj).comp = comp; %#ok<AGROW>
        diag_block.oracle_qr_sum = diag_block.oracle_qr_sum + comp.total;
    end
    diag_block.optimizer_sum = sum(arrayfun(@(x) x.comp.total, diag_block.optimizer));
end

function comp = combined_components(A_block, M_gain, old_row_memory, rows_seen, n, v)
    R = [old_row_memory; A_block];
    yg = M_gain * v;
    zp = R * v;
    yn = A_block * v;
    yo = old_row_memory * v;
    gain2 = kahan_sum(yg.^2);
    pooled_y2 = kahan_sum(zp.^2);
    pooled_y4 = kahan_sum(zp.^4);
    rows_entropy = size(R,1);
    c = log(rows_seen / n) / (2 * log(rows_entropy));
    pooled_H = -(log(max(pooled_y4, realmin)) - 2 * log(max(pooled_y2, realmin)));
    pooled_rel_H = pooled_H / log(rows_entropy);
    phi = exp(c * (log(max(pooled_y4, realmin)) - 2 * log(max(pooled_y2, realmin))));
    comp.total = gain2 * phi;
    comp.gain2 = gain2;
    comp.phi = phi;
    comp.pooled_y2 = pooled_y2;
    comp.pooled_y4 = pooled_y4;
    comp.pooled_H = pooled_H;
    comp.pooled_rel_H = pooled_rel_H;
    comp.combined_c = c;
    comp.rows_entropy = rows_entropy;
    comp.rows_seen = rows_seen;
    comp.new_y2 = kahan_sum(yn.^2);
    comp.new_y4 = kahan_sum(yn.^4);
    comp.new_H = entropy_from_y(yn);
    comp.new_rel_H = comp.new_H / log(size(A_block,1));
    comp.old_y2 = kahan_sum(yo.^2);
    comp.old_y4 = kahan_sum(yo.^4);
    comp.old_H = entropy_from_y(yo);
    if isempty(yo)
        comp.old_rel_H = nan;
    else
        comp.old_rel_H = comp.old_H / log(size(old_row_memory,1));
    end
    comp.old_rows = size(old_row_memory,1);
end

function diag_block = build_forget_diag(A_block, M_gain, V_out, V_exact, n, prev_basis, prev_s2, is_initial)
    diag_block = struct();
    for jj = 1:size(V_out,2)
        vraw = M_gain' * (M_gain * V_exact(:, jj));
        if norm(vraw) > 1e-14
            vraw = vraw / norm(vraw);
        end
        diag_block.oracle_raw(jj).v = vraw; %#ok<AGROW>
        diag_block.optimizer(jj).v = V_out(:, jj); %#ok<AGROW>
        if is_initial
            diag_block.oracle_raw(jj).comp = forget_components_initial(A_block, vraw, n); %#ok<AGROW>
            diag_block.optimizer(jj).comp = forget_components_initial(A_block, V_out(:, jj), n); %#ok<AGROW>
        else
            diag_block.oracle_raw(jj).comp = forget_components_streaming(A_block, prev_basis, prev_s2, vraw, n); %#ok<AGROW>
            diag_block.optimizer(jj).comp = forget_components_streaming(A_block, prev_basis, prev_s2, V_out(:, jj), n); %#ok<AGROW>
        end
    end
end

function comp = forget_components_initial(A_block, v, n)
    [score, ~, y2, H] = forget_score_grad_initial(A_block, v, n);
    y = A_block * v;
    comp.total = score;
    comp.old_E = 0;
    comp.new_E = y2;
    comp.new_H = H;
    comp.new_rel_H = H / log(size(A_block,1));
    comp.new_y4 = kahan_sum(y.^4);
end

function comp = forget_components_streaming(A_block, V_old, s2_old, v, n)
    [score, ~, P_total, H] = forget_score_grad_streaming(A_block, V_old, s2_old, v, n);
    a = V_old' * v;
    y = A_block * v;
    comp.total = score;
    comp.old_E = sum((a.^2) .* s2_old);
    comp.new_E = kahan_sum(y.^2);
    comp.P_total = P_total;
    comp.new_H = H;
    comp.new_rel_H = H / log(size(A_block,1));
    comp.new_y4 = kahan_sum(y.^4);
end

function print_combined_diag(diag_block)
    fprintf('combined_score_components:\n');
    for jj = 1:numel(diag_block.oracle_raw)
        c = diag_block.oracle_raw(jj).comp;
        fprintf('  oracle_raw_v%d: total=%.12g gain2=%.12g phi=%.12g pooled_rel_H=%.12g combined_c=%.12g new_rel_H=%.12g old_rel_H=%.12g old_rows=%d\n', ...
            jj, c.total, c.gain2, c.phi, c.pooled_rel_H, c.combined_c, c.new_rel_H, c.old_rel_H, c.old_rows);
    end
    for jj = 1:numel(diag_block.optimizer)
        c = diag_block.optimizer(jj).comp;
        fprintf('  optimizer_v%d:  total=%.12g gain2=%.12g phi=%.12g pooled_rel_H=%.12g combined_c=%.12g new_rel_H=%.12g old_rel_H=%.12g old_rows=%d\n', ...
            jj, c.total, c.gain2, c.phi, c.pooled_rel_H, c.combined_c, c.new_rel_H, c.old_rel_H, c.old_rows);
    end
    fprintf('oracle_qr_projection_scores sum=%.12g\n', diag_block.oracle_qr_sum);
    fprintf('optimizer_score_sum=%.12g\n', diag_block.optimizer_sum);
end

function print_forget_diag(diag_block)
    fprintf('forget_score_components:\n');
    for jj = 1:numel(diag_block.oracle_raw)
        c = diag_block.oracle_raw(jj).comp;
        fprintf('  oracle_raw_v%d: total=%.12g old_E=%.12g new_E=%.12g new_rel_H=%.12g\n', ...
            jj, c.total, c.old_E, c.new_E, c.new_rel_H);
    end
    for jj = 1:numel(diag_block.optimizer)
        c = diag_block.optimizer(jj).comp;
        fprintf('  optimizer_v%d:  total=%.12g old_E=%.12g new_E=%.12g new_rel_H=%.12g\n', ...
            jj, c.total, c.old_E, c.new_E, c.new_rel_H);
    end
end

function print_projection_summary(V_hat, V_exact, M_gain)
    Qcols = zeros(size(V_hat,1), size(V_hat,2));
    for jj = 1:size(V_hat,2)
        q = M_gain' * (M_gain * V_exact(:, jj));
        if norm(q) > 1e-14
            q = q / norm(q);
        end
        Qcols(:, jj) = q;
    end
    [Q_oracle, ~] = qr(Qcols, 0);
    fprintf('vecnorm(V_score V_score'' oracle_raw): ');
    disp(vecnorm(V_hat * V_hat' * Qcols, 2));
    fprintf('principal_cosines(V_score, Q_oracle): ');
    disp(svd(V_hat' * Q_oracle).');
end

function diag = build_tail_conspiracy_diag( ...
    block_id, rows, A_block, M_actual, M_zero, M_oracle, old_row_memory, rows_seen, n, ...
    V_selected, V_carried, V_exact, prev_selected, prev_carried)

    r = size(V_selected, 2);
    Q_exact = orth_cols(V_exact(:, 1:r));
    Q_oracle = projected_oracle_frame(M_actual, V_exact, r);

    sel_cos = principal_cosines(V_selected, Q_oracle);
    car_cos = principal_cosines(V_carried, Q_oracle);
    sel_tail_mass = frame_tail_mass(V_selected, Q_exact);
    car_tail_mass = frame_tail_mass(V_carried, Q_exact);

    opt1 = V_selected(:, 1);
    opt2 = V_selected(:, 2);
    carried1 = V_carried(:, 1);
    carried2 = V_carried(:, 2);
    opt1_tail = tail_component(opt1, Q_exact);
    opt2_tail = tail_component(opt2, Q_exact);
    carried1_tail = tail_component(carried1, Q_exact);
    carried2_tail = tail_component(carried2, Q_exact);

    diag = struct();
    diag.block_id = block_id;
    diag.rows = rows;
    diag.sel_oracle_cos1 = sel_cos(1);
    diag.sel_oracle_cos2 = sel_cos(min(2, numel(sel_cos)));
    diag.car_oracle_cos1 = car_cos(1);
    diag.car_oracle_cos2 = car_cos(min(2, numel(car_cos)));
    diag.sel_tail_mass = sel_tail_mass;
    diag.car_tail_mass = car_tail_mass;

    diag.opt1_tail_mass = vector_tail_mass(opt1, Q_exact);
    diag.opt2_tail_mass = vector_tail_mass(opt2, Q_exact);
    diag.carried1_tail_mass = vector_tail_mass(carried1, Q_exact);
    diag.carried2_tail_mass = vector_tail_mass(carried2, Q_exact);

    diag.opt1_vs_oracle1 = abs(real(opt1' * Q_oracle(:, 1)));
    diag.opt1_vs_oracle2 = abs(real(opt1' * Q_oracle(:, 2)));
    diag.opt2_vs_oracle1 = abs(real(opt2' * Q_oracle(:, 1)));
    diag.opt2_vs_oracle2 = abs(real(opt2' * Q_oracle(:, 2)));
    diag.carried1_vs_oracle1 = abs(real(carried1' * Q_oracle(:, 1)));
    diag.carried1_vs_oracle2 = abs(real(carried1' * Q_oracle(:, 2)));
    diag.carried2_vs_oracle1 = abs(real(carried2' * Q_oracle(:, 1)));
    diag.carried2_vs_oracle2 = abs(real(carried2' * Q_oracle(:, 2)));

    if isempty(prev_selected)
        diag.prev_opt1_dot_opt1 = nan;
        diag.prev_opt2_dot_opt2 = nan;
        diag.prev_opt1_tail_cos = nan;
        diag.prev_opt2_tail_cos = nan;
    else
        prev_opt1 = prev_selected(:, 1);
        prev_opt2 = prev_selected(:, 2);
        prev_opt1_tail = tail_component(prev_opt1, Q_exact);
        prev_opt2_tail = tail_component(prev_opt2, Q_exact);
        diag.prev_opt1_dot_opt1 = abs(real(prev_opt1' * opt1));
        diag.prev_opt2_dot_opt2 = abs(real(prev_opt2' * opt2));
        diag.prev_opt1_tail_cos = abs(real(prev_opt1_tail' * opt1_tail));
        diag.prev_opt2_tail_cos = abs(real(prev_opt2_tail' * opt2_tail));
    end

    if isempty(prev_carried)
        diag.prev_carried1_dot_carried1 = nan;
        diag.prev_carried2_dot_carried2 = nan;
        diag.prev_carried1_tail_cos = nan;
        diag.prev_carried2_tail_cos = nan;
    else
        prev_carried1 = prev_carried(:, 1);
        prev_carried2 = prev_carried(:, 2);
        prev_carried1_tail = tail_component(prev_carried1, Q_exact);
        prev_carried2_tail = tail_component(prev_carried2, Q_exact);
        diag.prev_carried1_dot_carried1 = abs(real(prev_carried1' * carried1));
        diag.prev_carried2_dot_carried2 = abs(real(prev_carried2' * carried2));
        diag.prev_carried1_tail_cos = abs(real(prev_carried1_tail' * carried1_tail));
        diag.prev_carried2_tail_cos = abs(real(prev_carried2_tail' * carried2_tail));
    end

    oracle2_vs_opt1 = retract_feasible(Q_oracle(:, 2), V_selected(:, 1));
    if isempty(oracle2_vs_opt1)
        oracle2_vs_opt1 = Q_oracle(:, 2);
    end
    oracle1 = Q_oracle(:, 1);
    oracle2 = Q_oracle(:, 2);

    opt1_actual = combined_components(A_block, M_actual, old_row_memory, rows_seen, n, opt1);
    opt1_zero = combined_components(A_block, M_zero, old_row_memory, rows_seen, n, opt1);
    opt1_oracle = combined_components(A_block, M_oracle, old_row_memory, rows_seen, n, opt1);
    opt2_actual = combined_components(A_block, M_actual, old_row_memory, rows_seen, n, opt2);
    opt2_zero = combined_components(A_block, M_zero, old_row_memory, rows_seen, n, opt2);
    opt2_oracle = combined_components(A_block, M_oracle, old_row_memory, rows_seen, n, opt2);

    oracle1_actual = combined_components(A_block, M_actual, old_row_memory, rows_seen, n, oracle1);
    oracle1_zero = combined_components(A_block, M_zero, old_row_memory, rows_seen, n, oracle1);
    oracle1_oracle = combined_components(A_block, M_oracle, old_row_memory, rows_seen, n, oracle1);
    oracle2_actual = combined_components(A_block, M_actual, old_row_memory, rows_seen, n, oracle2);
    oracle2_zero = combined_components(A_block, M_zero, old_row_memory, rows_seen, n, oracle2);
    oracle2_oracle = combined_components(A_block, M_oracle, old_row_memory, rows_seen, n, oracle2);

    oracle2_vs_opt1_actual = combined_components(A_block, M_actual, old_row_memory, rows_seen, n, oracle2_vs_opt1);
    oracle2_vs_opt1_zero = combined_components(A_block, M_zero, old_row_memory, rows_seen, n, oracle2_vs_opt1);
    oracle2_vs_opt1_oracle = combined_components(A_block, M_oracle, old_row_memory, rows_seen, n, oracle2_vs_opt1);

    diag.opt1_actual_score = opt1_actual.total;
    diag.opt1_zero_score = opt1_zero.total;
    diag.opt1_oracle_score = opt1_oracle.total;
    diag.opt1_actual_gain2 = opt1_actual.gain2;
    diag.opt1_zero_gain2 = opt1_zero.gain2;
    diag.opt1_oracle_gain2 = opt1_oracle.gain2;
    diag.opt1_relH = opt1_actual.pooled_rel_H;
    diag.opt2_actual_score = opt2_actual.total;
    diag.opt2_zero_score = opt2_zero.total;
    diag.opt2_oracle_score = opt2_oracle.total;
    diag.opt2_actual_gain2 = opt2_actual.gain2;
    diag.opt2_zero_gain2 = opt2_zero.gain2;
    diag.opt2_oracle_gain2 = opt2_oracle.gain2;
    diag.opt2_relH = opt2_actual.pooled_rel_H;

    diag.oracle1_actual_score = oracle1_actual.total;
    diag.oracle1_zero_score = oracle1_zero.total;
    diag.oracle1_oracle_score = oracle1_oracle.total;
    diag.oracle2_actual_score = oracle2_actual.total;
    diag.oracle2_zero_score = oracle2_zero.total;
    diag.oracle2_oracle_score = oracle2_oracle.total;
    diag.oracle2_vs_opt1_actual_score = oracle2_vs_opt1_actual.total;
    diag.oracle2_vs_opt1_zero_score = oracle2_vs_opt1_zero.total;
    diag.oracle2_vs_opt1_oracle_score = oracle2_vs_opt1_oracle.total;

    diag.opt1_actual_margin_vs_oracle1 = diag.opt1_actual_score - diag.oracle1_actual_score;
    diag.opt2_actual_margin_vs_oracle2 = diag.opt2_actual_score - diag.oracle2_actual_score;
    diag.opt2_actual_margin_vs_oracle2_opt1 = diag.opt2_actual_score - diag.oracle2_vs_opt1_actual_score;
    diag.opt2_zero_margin_vs_oracle2_opt1 = diag.opt2_zero_score - diag.oracle2_vs_opt1_zero_score;

    diag.opt1_sketch_boost_score = diag.opt1_actual_score - diag.opt1_zero_score;
    diag.opt2_sketch_boost_score = diag.opt2_actual_score - diag.opt2_zero_score;
    diag.opt1_sketch_boost_gain2 = diag.opt1_actual_gain2 - diag.opt1_zero_gain2;
    diag.opt2_sketch_boost_gain2 = diag.opt2_actual_gain2 - diag.opt2_zero_gain2;
    diag.opt1_sketch_score_share = safe_ratio(diag.opt1_sketch_boost_score, diag.opt1_actual_score);
    diag.opt2_sketch_score_share = safe_ratio(diag.opt2_sketch_boost_score, diag.opt2_actual_score);
    diag.opt1_sketch_gain_share = safe_ratio(diag.opt1_sketch_boost_gain2, diag.opt1_actual_gain2);
    diag.opt2_sketch_gain_share = safe_ratio(diag.opt2_sketch_boost_gain2, diag.opt2_actual_gain2);

    if isempty(prev_selected)
        diag.prev_opt1_actual_score = nan;
        diag.prev_opt1_score_ratio = nan;
        diag.prev_opt2_actual_score = nan;
        diag.prev_opt2_score_ratio = nan;
    else
        prev_comp1 = combined_components(A_block, M_actual, old_row_memory, rows_seen, n, prev_selected(:, 1));
        prev_comp2 = combined_components(A_block, M_actual, old_row_memory, rows_seen, n, prev_selected(:, 2));
        diag.prev_opt1_actual_score = prev_comp1.total;
        diag.prev_opt1_score_ratio = safe_ratio(prev_comp1.total, opt1_actual.total);
        diag.prev_opt2_actual_score = prev_comp2.total;
        diag.prev_opt2_score_ratio = safe_ratio(prev_comp2.total, opt2_actual.total);
    end

    fprintf('tail_conspiracy block=%d sel_oracle_cos=[%.6f %.6f] sel_tail=%.6f prev_opt2_dot=%.6f actual_opt2=%.6g zero_opt2=%.6g oracle_opt2=%.6g oracle2_vs_opt1_actual=%.6g sketch_share=%.3f\n', ...
        block_id, diag.sel_oracle_cos1, diag.sel_oracle_cos2, diag.sel_tail_mass, ...
        diag.prev_opt2_dot_opt2, diag.opt2_actual_score, diag.opt2_zero_score, ...
        diag.opt2_oracle_score, diag.oracle2_vs_opt1_actual_score, diag.opt2_sketch_score_share);
end

function plot_tail_conspiracy_results(hist, matrix_name, mode, do_save, fig_dir)
    blocks = field_vector(hist, 'block_id');

    fig = figure('Name', sprintf('Tail conspiracy: %s %s', matrix_name, mode), ...
        'Position', [100 100 1450 950]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    plot(blocks, field_vector(hist, 'opt1_vs_oracle1'), '-o', 'LineWidth', 1.2); hold on;
    plot(blocks, field_vector(hist, 'opt2_vs_oracle2'), '-o', 'LineWidth', 1.2);
    plot(blocks, field_vector(hist, 'opt1_tail_mass'), '--', 'LineWidth', 1.2);
    plot(blocks, field_vector(hist, 'opt2_tail_mass'), '--', 'LineWidth', 1.2);
    ylim([0 1.05]); grid on;
    xlabel('Block');
    ylabel('Cosine / mass');
    title('Both selected slots: oracle alignment and tail mass');
    legend({'opt1 vs oracle1', 'opt2 vs oracle2', 'opt1 tail mass', 'opt2 tail mass'}, ...
        'Location', 'southoutside');

    nexttile;
    plot(blocks, field_vector(hist, 'prev_opt1_tail_cos'), '-o', 'LineWidth', 1.2); hold on;
    plot(blocks, field_vector(hist, 'prev_opt2_tail_cos'), '-o', 'LineWidth', 1.2);
    plot(blocks, field_vector(hist, 'prev_opt1_score_ratio'), '--', 'LineWidth', 1.2);
    plot(blocks, field_vector(hist, 'prev_opt2_score_ratio'), '--', 'LineWidth', 1.2);
    ylim([0 1.05]);
    ylabel('Cosine / score ratio');
    grid on;
    xlabel('Block');
    title('Per-block slot persistence: tail cosine and previous-score ratio');
    legend({'prev opt1 tail cos', 'prev opt2 tail cos', 'prev opt1 score ratio', 'prev opt2 score ratio'}, ...
        'Location', 'southoutside');

    nexttile;
    plot(blocks, field_vector(hist, 'opt1_actual_score'), '-o', 'LineWidth', 1.2); hold on;
    plot(blocks, field_vector(hist, 'opt1_zero_score'), '-o', 'LineWidth', 1.2);
    plot(blocks, field_vector(hist, 'oracle1_actual_score'), '--', 'LineWidth', 1.2);
    plot(blocks, field_vector(hist, 'opt2_actual_score'), '-s', 'LineWidth', 1.2);
    plot(blocks, field_vector(hist, 'opt2_zero_score'), '-s', 'LineWidth', 1.2);
    plot(blocks, field_vector(hist, 'oracle2_vs_opt1_actual_score'), '--', 'LineWidth', 1.2);
    grid on;
    xlabel('Block');
    ylabel('Candidate score');
    title('Both slots: actual B versus zero B, plus oracle candidates');
    legend({'opt1 actual B', 'opt1 zero B', 'oracle1 actual B', ...
            'opt2 actual B', 'opt2 zero B', 'oracle2 vs opt1 actual B'}, ...
        'Location', 'southoutside');

    nexttile;
    plot(blocks, field_vector(hist, 'opt1_sketch_score_share'), '-o', 'LineWidth', 1.2); hold on;
    plot(blocks, field_vector(hist, 'opt2_sketch_score_share'), '-o', 'LineWidth', 1.2);
    plot(blocks, field_vector(hist, 'opt1_actual_margin_vs_oracle1'), '--', 'LineWidth', 1.2);
    plot(blocks, field_vector(hist, 'opt2_actual_margin_vs_oracle2_opt1'), '--', 'LineWidth', 1.2);
    yline(0, 'k:');
    grid on;
    xlabel('Block');
    ylabel('Share / margin');
    title('Sketch contribution and opt-vs-oracle margins for both slots');
    legend({'opt1 sketch score share', 'opt2 sketch score share', ...
            'opt1 actual - oracle1 actual', 'opt2 actual - oracle2-vs-opt1 actual'}, ...
        'Location', 'southoutside');

    sgtitle(sprintf('%s, %s: tail conspiracy and sketch-energy dominance', matrix_name, mode), ...
        'Interpreter', 'none');

    if do_save
        if ~exist(fig_dir, 'dir')
            mkdir(fig_dir);
        end
        base = fullfile(fig_dir, sprintf('%s_%s_tail_conspiracy', char(matrix_name), char(mode)));
        savefig(fig, base + ".fig");
        exportgraphics(fig, base + ".png", 'Resolution', 180);
        fprintf('Saved tail-conspiracy plot to %s.fig and %s.png\n', base, base);
    end
end

function plot_kept_vs_thrown_decomposition( ...
        M_gain, V_hat, block_id, row_range, matrix_name, mode, r, do_save, fig_dir, fd_info)
    % Per-window snapshot: how this block's augmented matrix M_gain splits
    % into the part carried forward (Kept) and the part the sketch discards
    % (Thrown). V_hat is the rank-r right basis selected for this block.
    %   Kept   = M_gain * V_hat * V_hat'    (projection onto carried subspace)
    %   Thrown = M_gain - Kept              (orthogonal-complement residual)
    %
    % FD shrinkage extension: for mode=="FD" the sketch carried forward is
    % NOT M_kept but diag(s_new)*V_hat' with s_new = sqrt(max(s_top.^2-d,0)).
    % Two extra traces are overlaid:
    %   - Kept panel: s_new (FD actual) alongside s_top (geometric kept).
    %   - Thrown panel: per-direction shrinkage residual sqrt(s_top.^2-s_new.^2)
    %     and a horizontal sqrt(delta) reference (uniform FD margin per kept dir).
    if nargin < 10, fd_info = []; end

    r_use = size(V_hat, 2);
    M_kept   = M_gain * V_hat * V_hat';
    M_thrown = M_gain - M_kept;

    s_kept   = svd(M_kept);
    s_thrown = svd(M_thrown);
    % Drop the structural zeros (rank(M_kept) <= r, rank(M_thrown) bounded
    % above by min(size(M_gain)) - r), so the log axis stays informative.
    floor_kept   = max(s_kept)   * 1e-14 + eps;
    floor_thrown = max(s_thrown) * 1e-14 + eps;
    s_kept   = s_kept(s_kept     > floor_kept);
    s_thrown = s_thrown(s_thrown > floor_thrown);

    energy_total      = sum(M_gain(:).^2);
    energy_kept_geom  = sum(M_kept(:).^2);
    energy_thrown_geom = max(energy_total - energy_kept_geom, 0);
    frac_kept_geom    = energy_kept_geom    / max(energy_total, eps);
    frac_thrown_geom  = energy_thrown_geom / max(energy_total, eps);

    is_fd = ~isempty(fd_info) && isfield(fd_info, 'mode') && strcmp(fd_info.mode, "FD") ...
            && isfield(fd_info, 's_top') && isfield(fd_info, 's_new');
    if is_fd
        s_top = fd_info.s_top(:);
        s_new = fd_info.s_new(:);
        delta = fd_info.delta;
        shrink_resid = sqrt(max(s_top.^2 - s_new.^2, 0));   % per kept direction
        energy_kept_fd    = sum(s_new.^2);
        energy_shrink_fd  = sum(s_top.^2 - s_new.^2);       % >= 0
        frac_kept_fd      = energy_kept_fd   / max(energy_total, eps);
        frac_shrink_fd    = energy_shrink_fd / max(energy_total, eps);
    end

    fig = figure( ...
        'Name', sprintf('Kept vs thrown @ block %d: %s/%s (r=%d)', ...
            block_id, char(matrix_name), char(mode), r_use), ...
        'Color', 'w', 'Position', [100, 100, 1200, 480]);

    subplot(1, 2, 1);
    if isempty(s_kept), s_kept = nan; end
    semilogy(1:numel(s_kept), s_kept, 'o-', 'LineWidth', 1.4, 'MarkerSize', 6, ...
        'DisplayName', 'geometric kept (s\_top)');
    grid on; hold on;
    if is_fd
        semilogy(1:numel(s_new), s_new, 's--', 'LineWidth', 1.4, 'MarkerSize', 6, ...
            'Color', [0.10 0.55 0.20], 'DisplayName', 'FD actual (s\_new)');
        title(sprintf('Kept: geom %.4f, FD %.4f of ||M||_F^2', frac_kept_geom, frac_kept_fd));
        legend('Location', 'best');
    else
        title(sprintf('Kept = M_{gain} V_r V_r''  (%.4f of ||M||_F^2)', frac_kept_geom));
    end
    hold off;
    xlabel('index'); ylabel('singular value (log)');

    subplot(1, 2, 2);
    if isempty(s_thrown), s_thrown = nan; end
    semilogy(1:numel(s_thrown), s_thrown, 'x-', 'LineWidth', 1.4, 'MarkerSize', 6, ...
        'Color', [0.85 0.33 0.10], 'DisplayName', 'geometric thrown');
    grid on; hold on;
    if is_fd
        nz = shrink_resid > 0;
        if any(nz)
            semilogy(find(nz), shrink_resid(nz), 'd:', 'LineWidth', 1.4, 'MarkerSize', 6, ...
                'Color', [0.45 0.20 0.55], 'DisplayName', 'FD shrinkage residual');
        end
        if delta > 0
            yline(sqrt(delta), '--', sprintf('\\surd\\delta = %.3g', sqrt(delta)), ...
                'Color', [0.40 0.40 0.40], 'LineWidth', 1.0, ...
                'LabelHorizontalAlignment', 'left', 'DisplayName', '\surd\delta');
        end
        title(sprintf('Thrown: geom %.4f + FD shrink %.4f of ||M||_F^2', ...
            frac_thrown_geom, frac_shrink_fd));
        legend('Location', 'best');
    else
        title(sprintf('Thrown = M_{gain} - Kept  (%.4f of ||M||_F^2)', frac_thrown_geom));
    end
    hold off;
    xlabel('index'); ylabel('singular value (log)');

    sgtitle(sprintf('%s, %s @ block %d (rows %d:%d), r=%d', ...
        char(matrix_name), char(mode), block_id, row_range(1), row_range(2), r_use), ...
        'FontWeight', 'bold');

    if do_save
        if ~isfolder(fig_dir)
            mkdir(fig_dir);
        end
        base = fullfile(fig_dir, sprintf('%s_%s_r%d_block%02d_kept_vs_thrown', ...
            char(matrix_name), char(mode), r_use, block_id));
        savefig(fig, base + ".fig");
        exportgraphics(fig, base + ".png", 'Resolution', 180);
        fprintf('Saved kept-vs-thrown plot (block %d) to %s.fig and %s.png\n', ...
            block_id, base, base);
    end
end

function [V_proj, s_proj] = projected_subspace_svd(M_gain, V_basis)
    [Qb, ~] = qr(V_basis, 0);
    C = M_gain * Qb;
    [~, Ssmall, Vsmall] = svd(C, "econ");
    V_proj = Qb * Vsmall(:, 1:size(Qb,2));
    s_proj = diag(Ssmall);
end

function Q = projected_oracle_frame(M_gain, V_exact, r)
    d = size(V_exact, 1);
    Qcols = zeros(d, r);
    for jj = 1:r
        q = M_gain' * (M_gain * V_exact(:, jj));
        if norm(q) > 1e-14
            q = q / norm(q);
        else
            q = V_exact(:, jj);
        end
        Qcols(:, jj) = q;
    end
    [Q, ~] = qr(Qcols, 0);
end

function Q = orth_cols(X)
    if isempty(X)
        Q = X;
        return;
    end
    [Q, R] = qr(X, 0);
    keep = abs(diag(R)) > 1e-12;
    Q = Q(:, keep);
end

function c = principal_cosines(A, B)
    if isempty(A) || isempty(B)
        c = nan(1, min(size(A,2), size(B,2)));
        return;
    end
    Aq = orth_cols(A);
    Bq = orth_cols(B);
    c = svd(Aq' * Bq).';
end

function m = frame_tail_mass(V_frame, Q_exact)
    Vq = orth_cols(V_frame);
    if isempty(Vq) || isempty(Q_exact)
        m = nan;
        return;
    end
    sig_mass = norm(Q_exact * (Q_exact' * Vq), 'fro')^2 / max(size(Vq,2), 1);
    m = max(0, 1 - sig_mass);
end

function m = vector_tail_mass(v, Q_exact)
    if isempty(v) || isempty(Q_exact)
        m = nan;
        return;
    end
    nv = norm(v);
    if nv <= 1e-14
        m = nan;
        return;
    end
    v = v / nv;
    sig_mass = norm(Q_exact * (Q_exact' * v))^2;
    m = max(0, 1 - sig_mass);
end

function t = tail_component(v, Q_exact)
    t = v - Q_exact * (Q_exact' * v);
    nt = norm(t);
    if nt <= 1e-14
        t = zeros(size(v));
    else
        t = t / nt;
    end
end

function r = safe_ratio(num, den)
    if abs(den) <= 1e-14
        r = nan;
    else
        r = num / den;
    end
end

function vals = field_vector(hist, name)
    vals = nan(numel(hist), 1);
    for ii = 1:numel(hist)
        if isfield(hist(ii), name)
            vals(ii) = hist(ii).(name);
        end
    end
end

function H = entropy_from_y(y)
    if isempty(y)
        H = nan;
        return;
    end
    y2 = kahan_sum(y.^2);
    y4 = kahan_sum(y.^4);
    if y2 <= 1e-28 || y4 <= 1e-28
        H = inf;
    else
        H = -(log(y4) - 2 * log(y2));
    end
end

function g_feas = project_to_feasible_tangent(g, v, Q)
    g_feas = g - v * real(v' * g);
    if ~isempty(Q)
        g_feas = g_feas - Q * (Q' * g_feas);
    end
end

function x = project_feasible(x, Q)
    if isempty(x)
        return;
    end
    if ~isempty(Q)
        x = x - Q * (Q' * x);
    end
end

function v_new = retract_feasible(x, Q)
    if isempty(x)
        v_new = [];
        return;
    end
    x = project_feasible(x, Q);
    nx = norm(x);
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
    for ii = 1:numel(x)
        y = x(ii) - c;
        t = s + y;
        c = (t - s) - y;
        s = t;
    end
end

function Beta_w = augsvd_beta_compute(beta_mode, beta_state, A_seen, Omega_qqa, n)
    switch beta_mode
        case "reservoir"
            Beta_w = beta_state.buf;
        case "SA"
            if size(beta_state.SA, 1) > 0 && any(any(beta_state.SA ~= 0))
                Beta_w = beta_state.SA;
            else
                Beta_w = zeros(0, n);
            end
        case "QQA"
            if isempty(A_seen)
                Beta_w = zeros(0, n);
            else
                Y = A_seen * Omega_qqa';
                Q = orth(Y);
                Beta_w = Q' * A_seen;
            end
        otherwise
            error('Unknown beta_mode: %s', beta_mode);
    end
end

function beta_state = augsvd_beta_update(beta_mode, beta_state, A_block, k_beta, sketch_type, sketch_zeta)
    switch beta_mode
        case "reservoir"
            beta_state = augsvd_reservoir_update(beta_state, A_block, k_beta);
        case "SA"
            G = build_sketch_matrix(k_beta, size(A_block, 1), sketch_type, sketch_zeta);
            beta_state.SA = beta_state.SA + G * A_block;
        case "QQA"
            % No state to maintain — Omega is fixed at init and the
            % prototype recomputes Q from A_seen each window.
        otherwise
            error('Unknown beta_mode: %s', beta_mode);
    end
end

function beta_state = augsvd_reservoir_update(beta_state, A_block, k_beta)
    nb = size(A_block, 1);
    for i = 1:nb
        beta_state.rows_seen_total = beta_state.rows_seen_total + 1;
        t = beta_state.rows_seen_total;
        if size(beta_state.buf, 1) < k_beta
            beta_state.buf = [beta_state.buf; A_block(i, :)]; %#ok<AGROW>
        else
            j = randi(t);
            if j <= k_beta
                beta_state.buf(j, :) = A_block(i, :);
            end
        end
    end
end

function S = build_sketch_matrix(k, m, sketch_type, zeta)
% Build a k-by-m sketch matrix projecting m-dim inputs into k dims.
% Used by AugSVD's QQA (Omega_qqa = build_sketch_matrix(k_beta, n, ...))
% and SA (G = build_sketch_matrix(k_beta, rows_in_block, ...)) paths.
%
% sketch_type:
%   "gaussian"     - dense i.i.d. standard normal entries (matches legacy randn)
%   "sparse_sign"  - sparse matrix with zeta +/-1 nonzeros per column placed
%                    in distinct random rows; O(zeta * m) nnz total
%   "count_sketch" - sparse matrix with exactly 1 signed nonzero per column
%                    (each input coordinate hashed to one output row)
%   "hadamard"     - SRHT: k uniformly sampled rows of an m-by-m Hadamard
%                    matrix combined with random +/-1 column sign flips,
%                    scaled by 1/sqrt(m). Requires k <= m and m supported
%                    by MATLAB's hadamard().
%
% zeta : nonzeros per column for sparse_sign (ignored by other types).

    if nargin < 4 || isempty(zeta)
        zeta = 8;
    end

    switch sketch_type
        case "gaussian"
            S = randn(k, m);

        case "sparse_sign"
            zeta_eff = max(1, min(zeta, k));
            rows = zeros(zeta_eff, m);
            for j = 1:m
                rows(:, j) = randperm(k, zeta_eff)';
            end
            cols  = repmat(1:m, zeta_eff, 1);
            signs = randi(2, zeta_eff, m) * 2 - 3;   % +/-1
            S = sparse(rows(:), cols(:), signs(:), k, m);

        case "count_sketch"
            rows  = randi(k, 1, m);
            cols  = 1:m;
            signs = randi(2, 1, m) * 2 - 3;          % +/-1
            S = sparse(rows, cols, signs, k, m);

        case "hadamard"
            if k > m
                error('hadamard sketch requires k <= m (got k=%d, m=%d).', k, m);
            end
            H = hadamard(m);
            D = randi(2, 1, m) * 2 - 3;              % +/-1 column-sign flips
            sample_rows = randperm(m, k);
            S = bsxfun(@times, H(sample_rows, :), D) / sqrt(m);

        otherwise
            error('Unknown sketch_type: %s', sketch_type);
    end
end

