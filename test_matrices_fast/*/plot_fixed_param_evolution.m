clc;

% New file

%% =========================
% User settings
% ==========================
% ZIP_DIR = "D:\zips\small_entropy_zips";
% UNZIP_DIR = "unzipped_small_entropy_zips";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_1000_0.2236_entropyscore_fast";

ZIP_DIR = "D:\zips\small_isvd_zips";
UNZIP_DIR = "unzipped_small_isvd_zips";
FIG_DIR = "figures";
BASE_PREFIX = "kernel_stocks_1000_0.2236_isvd";

% ZIP_DIR = "D:\zips\small_ent_new_zips";
% UNZIP_DIR = "unzipped_small_ent_new_zips";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_1000_0.2236_entropyscore_fast";

% ZIP_DIR = "D:\zips\small_ent_expansion_zips";
% UNZIP_DIR = "unzipped_small_ent_expansion_zips";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_1000_0.2236_entropyscore_expansion";

SIZE = 110;
RESERVOIR_METHOD = "greedy";
TARGET_RANK = 10;
SCORE_RANK = [];

% FIX_MODE:
%   "window" -> hold W fixed, compare different ell values
%   "sketch" -> hold ell fixed, compare different W values
FIX_MODE = "window";
FIXED_VALUE = 124;

SHOW_BAND = true;
SORT_ASCENDING = true;

%% =========================
% Setup directories
% ==========================
if ~isfolder(ZIP_DIR)
    error("ZIP_DIR does not exist: %s", ZIP_DIR);
end

if ~isfolder(UNZIP_DIR)
    mkdir(UNZIP_DIR);
end

if ~isfolder(FIG_DIR)
    mkdir(FIG_DIR);
end

if ~isfolder(FIG_DIR + "_jpg")
    mkdir(FIG_DIR + "_jpg");
end

%% =========================
% Unzip all sampled zips
% ==========================
zip_files = dir(fullfile(ZIP_DIR, "*_sampled.zip"));
fprintf("Found %d zip files in %s\n", numel(zip_files), ZIP_DIR);

for zi = 1:numel(zip_files)
    zip_name = string(zip_files(zi).name);
    zip_path = fullfile(ZIP_DIR, zip_name);

    folder_name = erase(zip_name, "_sampled.zip");
    dest_dir = fullfile(UNZIP_DIR, folder_name);

    if isfolder(dest_dir)
        existing = dir(dest_dir);
        existing = existing(~ismember({existing.name}, {'.', '..'}));
        if ~isempty(existing)
            fprintf("Skipping unzip (already exists): %s\n", dest_dir);
            continue;
        end
    else
        mkdir(dest_dir);
    end

    fprintf("Unzipping %s -> %s\n", zip_path, dest_dir);
    unzip(zip_path, dest_dir);
end

%% =========================
% Scan unzipped experiment folders
% ==========================
all_dirs = dir(UNZIP_DIR);
all_dirs = all_dirs([all_dirs.isdir]);
all_dirs = all_dirs(~ismember({all_dirs.name}, {'.', '..'}));

groups = containers.Map('KeyType', 'char', 'ValueType', 'any');

for di = 1:numel(all_dirs)
    folder_name = string(all_dirs(di).name);
    parsed = local_parse_folder(folder_name);

    fprintf('\nChecking folder: %s\n', folder_name);
    if isempty(parsed)
        fprintf('  rejected: parse failed\n');
        continue;
    end

    fprintf('  parsed prefix=%s size=%d W=%d ell=%d sr=%s reservoir=%s seed=%d\n', ...
        parsed.prefix, parsed.size, parsed.window_size, parsed.sketch_size, ...
        local_score_rank_label(parsed.score_rank), parsed.reservoir, parsed.seed);
    if parsed.raw_prefix ~= parsed.prefix
        fprintf('  normalized raw prefix %s -> %s (variant=%s)\n', ...
            parsed.raw_prefix, parsed.prefix, parsed.prefix_variant);
    end

    if parsed.prefix ~= BASE_PREFIX
        fprintf('  rejected: prefix mismatch (wanted %s)\n', BASE_PREFIX);
        continue;
    end

    if parsed.reservoir ~= RESERVOIR_METHOD
        fprintf('  rejected: reservoir mismatch (wanted %s)\n', RESERVOIR_METHOD);
        continue;
    end

    if ~isempty(SCORE_RANK) && ~isequal(parsed.score_rank, SCORE_RANK)
        fprintf('  rejected: score_rank mismatch (wanted %d)\n', SCORE_RANK);
        continue;
    end

    if parsed.window_size + parsed.sketch_size ~= parsed.size
        fprintf('  rejected: W + ell = %d, but size = %d\n', ...
            parsed.window_size + parsed.sketch_size, parsed.size);
        continue;
    end

    key = sprintf('%d__%d__%s', parsed.window_size, parsed.sketch_size, ...
        local_score_rank_key(parsed.score_rank));

    entry = struct();
    entry.seed = parsed.seed;
    entry.folder = folder_name;
    entry.parsed = parsed;

    if ~isKey(groups, key)
        groups(key) = {entry};
    else
        tmp = groups(key);
        tmp{end+1} = entry;
        groups(key) = tmp;
    end
end

group_keys = groups.keys;
group_window_size = zeros(size(group_keys));
group_sketch_size = zeros(size(group_keys));
group_score_rank = nan(size(group_keys));

for gi = 1:numel(group_keys)
    parts = split(string(group_keys{gi}), "__");
    group_window_size(gi) = str2double(parts(1));
    group_sketch_size(gi) = str2double(parts(2));
    if numel(parts) >= 3 && parts(3) ~= "na"
        group_score_rank(gi) = str2double(parts(3));
    end
end

available_pairs = [group_window_size(:), group_sketch_size(:), group_score_rank(:)];
available_window_sizes = unique(sort(group_window_size(:)));
available_sketch_sizes = unique(sort(group_sketch_size(:)));

if FIX_MODE == "window"
    keep_mask = group_window_size == FIXED_VALUE & group_sketch_size >= TARGET_RANK;
    varying_values = group_sketch_size;
    available_values = available_window_sizes;
    fixed_label = "W";
    varying_label = "ell";
elseif FIX_MODE == "sketch"
    keep_mask = group_sketch_size == FIXED_VALUE;
    varying_values = group_window_size;
    available_values = available_sketch_sizes;
    fixed_label = "ell";
    varying_label = "W";
else
    error('Unknown FIX_MODE: %s', FIX_MODE);
end

group_keys = group_keys(keep_mask);
group_window_size = group_window_size(keep_mask);
group_sketch_size = group_sketch_size(keep_mask);
group_score_rank = group_score_rank(keep_mask);
varying_values = varying_values(keep_mask);

if isempty(group_keys)
    fprintf('Available %s values: [%s]\n', fixed_label, num2str(available_values.'));
    fprintf('Available (W, ell) pairs:\n');
    disp(available_pairs);

    if isempty(available_values)
        error('No experiment groups were parsed for BASE_PREFIX=%s and reservoir=%s', ...
            BASE_PREFIX, RESERVOIR_METHOD);
    end

    [~, nearest_idx] = min(abs(available_values - FIXED_VALUE));
    error(['No experiment groups matched FIX_MODE=%s and FIXED_VALUE=%d.\n' ...
           'Try %s=%d instead, or choose from [%s].'], ...
        FIX_MODE, FIXED_VALUE, fixed_label, available_values(nearest_idx), ...
        num2str(available_values.'));
end

fprintf('Selected fixed %s=%d; varying %s values: [%s]\n', ...
    fixed_label, FIXED_VALUE, varying_label, num2str(varying_values(:).'));

if SORT_ASCENDING
    [~, order] = sort(varying_values);
else
    [~, order] = sort(varying_values, 'descend');
end

group_keys = group_keys(order);
group_window_size = group_window_size(order);
group_sketch_size = group_sketch_size(order);
group_score_rank = group_score_rank(order);
varying_values = varying_values(order);

%% =========================
% Create figure
% ==========================
fig_handle = figure('Position', [100 100 2100 650]);

ax_trace = subplot(1,3,1);
hold(ax_trace, 'on');

ax_residual = subplot(1,3,2);
hold(ax_residual, 'on');

ax_score = subplot(1,3,3);
hold(ax_score, 'on');

plot_results = struct();
plot_results.groups = groups;
plot_results.fix_mode = FIX_MODE;
plot_results.fixed_value = FIXED_VALUE;
plot_results.target_rank = TARGET_RANK;
plot_results.trace = {};
plot_results.residual = {};
plot_results.score_history = {};

used_trace = false;
used_residual = false;
used_score = false;

%% =========================
% Main loop over selected groups
% ==========================
for idx = 1:numel(group_keys)
    key = group_keys{idx};
    W = group_window_size(idx);
    ell = group_sketch_size(idx);
    sr = group_score_rank(idx);

    seed_entries = groups(key);
    seed_values = zeros(1, numel(seed_entries));
    for j = 1:numel(seed_entries)
        seed_values(j) = seed_entries{j}.seed;
    end
    [~, seed_order] = sort(seed_values);
    seed_entries = seed_entries(seed_order);

    trace_curves = {};
    trace_iters = {};
    trace_x_rows = {};
    trace_seeds = [];

    residual_curves = {};
    residual_iters = {};
    residual_x_rows = {};
    residual_seeds = [];
    residual_source_kind_group = "";

    score_curves = {};
    score_iters = {};
    score_x_rows = {};
    score_seeds = [];
    score_source_kind_group = "";

    for j = 1:numel(seed_entries)
        seed = seed_entries{j}.seed;
        folder = seed_entries{j}.folder;
        exp_dir = fullfile(UNZIP_DIR, folder);

        [trace_curve, trace_iter] = local_load_trace_error_curve(exp_dir, TARGET_RANK);
        if ~isempty(trace_curve) && ~isempty(trace_iter)
            x_rows = local_load_x_from_window_info(exp_dir, trace_iter);
            if ~isempty(x_rows)
                trace_curves{end+1} = max(trace_curve(:).', eps); %#ok<AGROW>
                trace_iters{end+1} = trace_iter; %#ok<AGROW>
                trace_x_rows{end+1} = x_rows; %#ok<AGROW>
                trace_seeds(end+1) = seed; %#ok<AGROW>
            end
        end

        [res_curve, res_iter, res_source_kind] = local_load_residual_curve(exp_dir, TARGET_RANK);
        if ~isempty(res_curve) && ~isempty(res_iter)
            x_rows = local_load_x_from_window_info(exp_dir, res_iter);
            if ~isempty(x_rows)
                residual_curves{end+1} = max(res_curve(:).', eps); %#ok<AGROW>
                residual_iters{end+1} = res_iter; %#ok<AGROW>
                residual_x_rows{end+1} = x_rows; %#ok<AGROW>
                residual_seeds(end+1) = seed; %#ok<AGROW>
                if residual_source_kind_group == ""
                    residual_source_kind_group = res_source_kind;
                end
            end
        end

        [score_curve, score_iter, score_source_kind] = local_load_score_history_curve(exp_dir, TARGET_RANK);
        if ~isempty(score_curve) && ~isempty(score_iter)
            x_rows = local_load_x_from_window_info(exp_dir, score_iter);
            if ~isempty(x_rows)
                score_curves{end+1} = max(score_curve(:).', eps); %#ok<AGROW>
                score_iters{end+1} = score_iter; %#ok<AGROW>
                score_x_rows{end+1} = x_rows; %#ok<AGROW>
                score_seeds(end+1) = seed; %#ok<AGROW>
                if score_source_kind_group == ""
                    score_source_kind_group = score_source_kind;
                end
            end
        end
    end

    line_label = sprintf('W=%d, ell=%d%s', W, ell, local_score_rank_short(sr));

    if ~isempty(trace_curves)
        [mean_curve, low_curve, high_curve, common_iters, x_rows] = ...
            local_aggregate_seed_curves(trace_curves, trace_iters, trace_x_rows);

        if ~isempty(common_iters)
            x = x_rows;
            h = semilogy(ax_trace, x, mean_curve, '-o', ...
                'LineWidth', 1.5, ...
                'DisplayName', line_label);

            if SHOW_BAND && numel(trace_curves) > 1
                p = fill(ax_trace, [x, fliplr(x)], ...
                    [low_curve, fliplr(high_curve)], h.Color, ...
                    'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
                    'HandleVisibility', 'off');
                set(p, 'HitTest', 'off', 'PickableParts', 'none');
            end

            out = struct();
            out.window_size = W;
            out.sketch_size = ell;
            out.score_rank = sr;
            out.common_iters = common_iters;
            out.x = x;
            out.x_window = x;
            out.x_rows = x_rows;
            out.mean_curve = mean_curve;
            out.low_curve = low_curve;
            out.high_curve = high_curve;
            out.seeds_used = trace_seeds;
            plot_results.trace{end+1} = out;

            used_trace = true;
            fprintf('Trace curve used for (W=%d, ell=%d), seeds=[%s], npts=%d\n', ...
                W, ell, num2str(trace_seeds), numel(common_iters));
        end
    end

    if ~isempty(residual_curves)
        [mean_curve, low_curve, high_curve, common_iters, x_rows] = ...
            local_aggregate_seed_curves(residual_curves, residual_iters, residual_x_rows);

        if ~isempty(common_iters)
            x = x_rows;
            h = semilogy(ax_residual, x, mean_curve, '-o', ...
                'LineWidth', 1.5, ...
                'DisplayName', line_label);

            if SHOW_BAND && numel(residual_curves) > 1
                p = fill(ax_residual, [x, fliplr(x)], ...
                    [low_curve, fliplr(high_curve)], h.Color, ...
                    'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
                    'HandleVisibility', 'off');
                set(p, 'HitTest', 'off', 'PickableParts', 'none');
            end

            out = struct();
            out.window_size = W;
            out.sketch_size = ell;
            out.score_rank = sr;
            out.common_iters = common_iters;
            out.x = x;
            out.x_window = x;
            out.x_rows = x_rows;
            out.mean_curve = mean_curve;
            out.low_curve = low_curve;
            out.high_curve = high_curve;
            out.seeds_used = residual_seeds;
            out.source_kind = residual_source_kind_group;
            plot_results.residual{end+1} = out;

            used_residual = true;
            fprintf('Residual curve used for (W=%d, ell=%d), seeds=[%s], npts=%d\n', ...
                W, ell, num2str(residual_seeds), numel(common_iters));
        end
    end

    if ~isempty(score_curves)
        [mean_curve, low_curve, high_curve, common_iters, x_rows] = ...
            local_aggregate_seed_curves(score_curves, score_iters, score_x_rows);

        if ~isempty(common_iters)
            x = x_rows;
            h = semilogy(ax_score, x, mean_curve, '-o', ...
                'LineWidth', 1.5, ...
                'DisplayName', line_label);

            if SHOW_BAND && numel(score_curves) > 1
                p = fill(ax_score, [x, fliplr(x)], ...
                    [low_curve, fliplr(high_curve)], h.Color, ...
                    'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
                    'HandleVisibility', 'off');
                set(p, 'HitTest', 'off', 'PickableParts', 'none');
            end

            out = struct();
            out.window_size = W;
            out.sketch_size = ell;
            out.score_rank = sr;
            out.common_iters = common_iters;
            out.x = x;
            out.x_window = x;
            out.x_rows = x_rows;
            out.mean_curve = mean_curve;
            out.low_curve = low_curve;
            out.high_curve = high_curve;
            out.seeds_used = score_seeds;
            out.source_kind = score_source_kind_group;
            plot_results.score_history{end+1} = out;

            used_score = true;
            fprintf('Score-history curve used for (W=%d, ell=%d), seeds=[%s], npts=%d\n', ...
                W, ell, num2str(score_seeds), numel(common_iters));
        end
    end
end

%% =========================
% Axis styling
% ==========================
grid(ax_trace, 'on');
ax_trace.XMinorGrid = 'on';
ax_trace.YMinorGrid = 'on';
ax_trace.GridLineStyle = '--';
ax_trace.GridAlpha = 0.5;
yscale(ax_trace, 'log');
xlabel(ax_trace, 'Rows processed');
ylabel(ax_trace, sprintf('Relative trace error (top-%d)', TARGET_RANK));
title(ax_trace, sprintf('Trace error evolution, %s=%d', local_fixed_label(FIX_MODE), FIXED_VALUE));

grid(ax_residual, 'on');
ax_residual.XMinorGrid = 'on';
ax_residual.YMinorGrid = 'on';
ax_residual.GridLineStyle = '--';
ax_residual.GridAlpha = 0.5;
yscale(ax_residual, 'log');
xlabel(ax_residual, 'Rows processed');
ylabel(ax_residual, sprintf('Residual norm (top-%d)', TARGET_RANK));
title(ax_residual, sprintf('Residual evolution, %s=%d', local_fixed_label(FIX_MODE), FIXED_VALUE));

if used_trace
    legend(ax_trace, 'Location', 'best', 'FontSize', 8);
end

if used_residual
    legend(ax_residual, 'Location', 'best', 'FontSize', 8);
end

grid(ax_score, 'on');
ax_score.XMinorGrid = 'on';
ax_score.YMinorGrid = 'on';
ax_score.GridLineStyle = '--';
ax_score.GridAlpha = 0.5;
yscale(ax_score, 'log');
xlabel(ax_score, 'Rows processed');
ylabel(ax_score, sprintf('Score history summary (top-%d)', TARGET_RANK));
title(ax_score, sprintf('Score history evolution, %s=%d', local_fixed_label(FIX_MODE), FIXED_VALUE));

if used_score
    legend(ax_score, 'Location', 'best', 'FontSize', 8);
end

sgtitle(sprintf('%s%s, reservoir=%s, size=%d, fixed %s=%d', ...
    BASE_PREFIX, local_score_rank_title_suffix(SCORE_RANK), RESERVOIR_METHOD, SIZE, local_fixed_label(FIX_MODE), FIXED_VALUE), ...
    'Interpreter', 'none');

%% =========================
% Save figures
% ==========================
save_tag = sprintf('%s%s_fix_%s_%d_rank_%d', ...
    BASE_PREFIX, local_score_rank_file_suffix(SCORE_RANK), char(local_fixed_label(FIX_MODE)), FIXED_VALUE, TARGET_RANK);

fig_path = fullfile(FIG_DIR, save_tag + ".fig");
savefig(fig_handle, fig_path);
fprintf('\nSaved MATLAB figure to: %s\n', fig_path);

jpg_path = fullfile(FIG_DIR + "_jpg", save_tag + ".jpg");
tmp_png = fullfile(FIG_DIR + "_jpg", save_tag + "_tmp.png");
exportgraphics(fig_handle, tmp_png, 'Resolution', 200);
img = imread(tmp_png);
imwrite(img, jpg_path, 'jpg', 'Quality', 40);
delete(tmp_png);
fprintf('Saved JPEG figure to: %s\n', jpg_path);

%% =========================
% Local functions
% ==========================
function parsed = local_parse_folder(folder_name)
    parsed = [];

    folder_name = char(folder_name);

    expr_seeded = ['^(?<prefix>.+)_random_uniform_(?<seed>\d+)' ...
                   '_size_(?<size>\d+)' ...
                   '_ssize_(?<window_size>\d+)' ...
                   '_k_(?<sketch_size>\d+)' ...
                   '(?:_sr_(?<score_rank>\d+))?' ...
                   '_reservoir_(?<reservoir>.+)$'];

    m = regexp(folder_name, expr_seeded, 'names');
    if ~isempty(m)
        parsed = struct();
        [parsed.prefix, parsed.raw_prefix, parsed.prefix_variant] = local_normalize_prefix(string(m.prefix));
        parsed.seed = str2double(m.seed);
        parsed.size = str2double(m.size);
        parsed.window_size = str2double(m.window_size);
        parsed.sketch_size = str2double(m.sketch_size);
        parsed.score_rank = local_parse_score_rank(folder_name);
        parsed.reservoir = string(m.reservoir);
        return;
    end

    expr_unseeded = ['^(?<prefix>.+)_random_uniform' ...
                     '_size_(?<size>\d+)' ...
                     '_ssize_(?<window_size>\d+)' ...
                     '_k_(?<sketch_size>\d+)' ...
                     '(?:_sr_(?<score_rank>\d+))?' ...
                     '_reservoir_(?<reservoir>.+)$'];

    m = regexp(folder_name, expr_unseeded, 'names');
    if ~isempty(m)
        parsed = struct();
        [parsed.prefix, parsed.raw_prefix, parsed.prefix_variant] = local_normalize_prefix(string(m.prefix));
        parsed.seed = 1;
        parsed.size = str2double(m.size);
        parsed.window_size = str2double(m.window_size);
        parsed.sketch_size = str2double(m.sketch_size);
        parsed.score_rank = local_parse_score_rank(folder_name);
        parsed.reservoir = string(m.reservoir);
    end
end

function [normalized_prefix, raw_prefix, prefix_variant] = local_normalize_prefix(raw_prefix)
    normalized_prefix = string(raw_prefix);
    prefix_variant = "";

    known_suffixes = ["_svd_aux"];
    for ii = 1:numel(known_suffixes)
        suffix = known_suffixes(ii);
        if endsWith(normalized_prefix, suffix)
            normalized_prefix = extractBefore(normalized_prefix, strlength(normalized_prefix) - strlength(suffix) + 1);
            prefix_variant = extractAfter(suffix, 1);
            return;
        end
    end
end

function score_rank = local_parse_score_rank(folder_name)
    score_rank = NaN;
    m = regexp(folder_name, '_sr_(?<score_rank>\d+)_reservoir_', 'names', 'once');
    if ~isempty(m)
        score_rank = str2double(m.score_rank);
    end
end

function key = local_score_rank_key(score_rank)
    if isnan(score_rank)
        key = 'na';
    else
        key = sprintf('%d', score_rank);
    end
end

function out = local_score_rank_label(score_rank)
    if isnan(score_rank)
        out = "none";
    else
        out = string(score_rank);
    end
end

function out = local_score_rank_short(score_rank)
    if isnan(score_rank)
        out = "";
    else
        out = sprintf(', sr=%d', score_rank);
    end
end

function out = local_score_rank_file_suffix(score_rank)
    if isempty(score_rank)
        out = "";
    else
        out = sprintf('_sr_%d', score_rank);
    end
end

function out = local_score_rank_title_suffix(score_rank)
    if isempty(score_rank)
        out = "";
    else
        out = sprintf(', sr=%d', score_rank);
    end
end

function out = local_load_txt(filename)
    try
        txt = fileread(filename);
        raw = jsondecode(txt);
    catch
        fprintf('Skipping invalid JSON file: %s\n', filename);
        out = struct();
        return;
    end

    out = struct();
    fns = fieldnames(raw);
    for ii = 1:numel(fns)
        key = fns{ii};
        obj = raw.(key);
        if isstruct(obj) && isfield(obj, 'type') && isfield(obj, 'value')
            out.(key) = obj.value;
        else
            out.(key) = obj;
        end
    end
end

function path = local_find_txt_file(exp_dir, prefix, iteration)
    candidate = fullfile(exp_dir, sprintf('%s_%d.txt', prefix, iteration));
    if isfile(candidate)
        path = candidate;
    else
        path = "";
    end
end

function iters = local_list_consecutive_iterations(exp_dir, prefix)
    txt_files = dir(fullfile(exp_dir, sprintf('%s_*.txt', prefix)));
    if isempty(txt_files)
        iters = [];
        return;
    end

    nums = [];
    expr = ['^' regexptranslate('escape', char(prefix)) '_(?<iter>\d+)\.txt$'];
    for ii = 1:numel(txt_files)
        m = regexp(txt_files(ii).name, expr, 'names');
        if ~isempty(m)
            nums(end+1) = str2double(m.iter); %#ok<AGROW>
        end
    end

    iters = unique(sort(nums));
end

function [curve, used_iters] = local_load_trace_error_curve(exp_dir, rank_limit)
    curve = [];
    used_iters = [];

    spectrum_iters = local_list_consecutive_iterations(exp_dir, "spectrum_data");
    if isempty(spectrum_iters)
        return;
    end

    S_exact = [];
    preferred_path = local_find_txt_file(exp_dir, "spectrum_data", 0);

    if preferred_path ~= ""
        data0 = local_load_txt(preferred_path);
        if isfield(data0, 'S_exact')
            S_exact = double(data0.S_exact(:));
        end
    end

    if isempty(S_exact)
        for ii = 1:numel(spectrum_iters)
            path = local_find_txt_file(exp_dir, "spectrum_data", spectrum_iters(ii));
            if path == ""
                continue;
            end
            data = local_load_txt(path);
            if isfield(data, 'S_exact')
                S_exact = double(data.S_exact(:));
                break;
            end
        end
    end

    if isempty(S_exact) || S_exact(1) == 0
        return;
    end

    S_exact_norm = S_exact ./ S_exact(1);

    vals = [];
    for ii = 1:numel(spectrum_iters)
        j = spectrum_iters(ii);
        path = local_find_txt_file(exp_dir, "spectrum_data", j);
        if path == ""
            continue;
        end

        data = local_load_txt(path);
        if ~isfield(data, 'S')
            continue;
        end

        S = double(data.S(:));
        rr = min([rank_limit, numel(S), numel(S_exact_norm)]);
        if rr <= 0
            continue;
        end

        denom = sum(S_exact_norm(1:rr));
        if denom == 0
            continue;
        end

        vals(end+1) = abs(denom - sum(S(1:rr))) / abs(denom); %#ok<AGROW>
        used_iters(end+1) = j; %#ok<AGROW>
    end

    curve = vals(:).';
end

function [curve, used_iters, source_kind] = local_load_residual_curve(exp_dir, rank_limit)
    curve = [];
    used_iters = [];
    source_kind = "";

    residual_sources = { ...
        struct('prefix', "reservoir_residuals_data", 'field', 'regular_residuals', 'label', "regular_residuals"), ...
        struct('prefix', "residuals_sym_psd_data", 'field', 'approx_residuals', 'label', "approx_residuals"), ...
        struct('prefix', "residuals_sym_psd_data_truncated", 'field', 'approx_residuals', 'label', "approx_residuals"), ...
        struct('prefix', "residuals_sym_psd_data_truncated_Rayleigh", 'field', 'approx_residuals', 'label', "approx_residuals") ...
    };

    for si = 1:numel(residual_sources)
        residual_iters = local_list_consecutive_iterations(exp_dir, residual_sources{si}.prefix);
        if isempty(residual_iters)
            continue;
        end

        vals = [];
        iters = [];
        for ii = 1:numel(residual_iters)
            j = residual_iters(ii);
            path = local_find_txt_file(exp_dir, residual_sources{si}.prefix, j);
            if path == ""
                continue;
            end

            data = local_load_txt(path);
            if ~isfield(data, residual_sources{si}.field)
                continue;
            end

            r = double(data.(residual_sources{si}.field)(:));
            rr = min(rank_limit, numel(r));
            if rr <= 0
                continue;
            end

            vals(end+1) = norm(r(1:rr), 2); %#ok<AGROW>
            iters(end+1) = j; %#ok<AGROW>
        end

        if ~isempty(vals)
            curve = vals(:).';
            used_iters = iters(:).';
            source_kind = sprintf('sqrt(sum_{i<=%d} %s(i)^2)', rank_limit, residual_sources{si}.label);
            return;
        end
    end
end

function [curve, used_iters, source_kind] = local_load_score_history_curve(exp_dir, rank_limit)
    curve = [];
    used_iters = [];
    source_kind = "";

    spectrum_iters = local_list_consecutive_iterations(exp_dir, "spectrum_data");
    if isempty(spectrum_iters)
        return;
    end

    vals = [];
    for ii = 1:numel(spectrum_iters)
        j = spectrum_iters(ii);
        path = local_find_txt_file(exp_dir, "spectrum_data", j);
        if path == ""
            continue;
        end

        data = local_load_txt(path);
        if ~isfield(data, 'score_history')
            continue;
        end

        s = double(data.score_history(:));
        rr = min(rank_limit, numel(s));
        if rr <= 0
            continue;
        end

        vals(end+1) = sum(max(s(1:rr), 0)); %#ok<AGROW>
        used_iters(end+1) = j; %#ok<AGROW>
    end

    if ~isempty(vals)
        curve = vals(:).';
        source_kind = sprintf('sum(score_history(1:%d))', rank_limit);
    end
end

function x = local_load_x_from_window_info(exp_dir, iters)
    x = [];
    if isempty(iters)
        return;
    end

    vals = nan(1, numel(iters));
    for ii = 1:numel(iters)
        path = local_find_txt_file(exp_dir, "window_info", iters(ii));
        if path == ""
            return;
        end

        data = local_load_txt(path);
        if ~isfield(data, 'end_idx')
            return;
        end

        vals(ii) = double(data.end_idx);
    end

    x = vals;
end

function [mean_curve, low_curve, high_curve, common_iters, common_x_rows] = ...
    local_aggregate_seed_curves(curves, iter_lists, x_rows_lists)

    mean_curve = [];
    low_curve = [];
    high_curve = [];
    common_iters = [];
    common_x_rows = [];

    if isempty(curves)
        return;
    end

    common_iters = iter_lists{1};
    for ii = 2:numel(iter_lists)
        common_iters = intersect(common_iters, iter_lists{ii});
    end
    common_iters = sort(common_iters);

    if isempty(common_iters)
        return;
    end

    arr_log10 = zeros(numel(curves), numel(common_iters));
    common_x_rows = nan(1, numel(common_iters));

    for ii = 1:numel(curves)
        [tf, loc] = ismember(common_iters, iter_lists{ii});
        if ~all(tf)
            mean_curve = [];
            low_curve = [];
            high_curve = [];
            common_iters = [];
            common_x_rows = [];
            return;
        end

        vals = curves{ii}(loc);
        arr_log10(ii, :) = log10(vals);

        xi = x_rows_lists{ii}(loc);
        if ii == 1
            common_x_rows = xi;
        end
    end

    mean_curve = 10.^mean(arr_log10, 1);
    low_curve = 10.^min(arr_log10, [], 1);
    high_curve = 10.^max(arr_log10, [], 1);
end

function out = local_fixed_label(fix_mode)
    if fix_mode == "window"
        out = "W";
    else
        out = "ell";
    end
end
