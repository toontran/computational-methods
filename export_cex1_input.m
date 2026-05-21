clear; clc;
if exist('rng', 'file') || exist('rng', 'builtin')
    rng(0);
else
    rand('seed', 0);
    randn('seed', 0);
end

n = 1024;
r_sig = 2;
alpha_sig = 0.003;
alpha_tail = 0.0145;
tail_scale = 0.99;
sigma1 = 0.991;

k = n;
U0 = zeros(n, n);

if mod(n, 2) ~= 0
    error('Hadamard construction requires n to be even.');
end

H = hadamard(n);
U0(:, 1:r_sig) = H(:, 1:r_sig) / sqrt(n);

a_tail = sqrt(1 - r_sig / n);
b_tail = 1 / sqrt(n);
for j = r_sig + 1:n
    col = zeros(n, 1);
    idx_large = j - r_sig;
    col(idx_large) = a_tail;
    col(n-r_sig+1:n) = b_tail;
    U0(:, j) = col;
end

if exist('OCTAVE_VERSION', 'builtin')
    [U, ~, ~] = svd(U0, 'econ');
else
    [U, ~] = qr(U0, 0);
end
for j = 1:r_sig
    if dot(U(:, j), U0(:, j)) < 0
        U(:, j) = -U(:, j);
    end
end

[V, ~] = qr(randn(n, k), 0);

sig_block = sigma1 * (1:r_sig).^(-alpha_sig);
tail_block = tail_scale * (1:(k-r_sig)).^(-alpha_tail);
svec = [sig_block, tail_block];
svec(1) = sigma1;

p = randperm(n);
A_unpermuted = U * diag(svec) * V';
A = A_unpermuted(p, :);
U_perm = U(p, :);

out_file = fullfile('matlab', 'cex1_input.mat');
save(out_file, 'A', 'A_unpermuted', 'U', 'U_perm', 'V', 'svec', 'p', ...
    'n', 'r_sig', 'alpha_sig', 'alpha_tail', 'tail_scale', 'sigma1', '-v7');
fprintf('Saved %s\n', out_file);
