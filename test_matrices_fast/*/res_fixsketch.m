clc;

% Old file


%% =========================
% User settings
% ==========================
% ZIP_DIR = "C:\Users\Admin\sampled_zips";
% ZIP_DIR = "C:\Users\Admin\fix_sketch_new_zips";
ZIP_DIR = "C:\Users\Admin\badcase_n_missing_cases_zips";

% UNZIP_DIR = "unzipped_kernel_win_v_stream";
% UNZIP_DIR = "unzipped_sketch1";
UNZIP_DIR = "unzipped_missing1";
FIG_DIR = "figures_matlab_2";

ZIP_DIR = "C:\Users\Admin\fix_res_zips";
UNZIP_DIR = "unzipped_fixres";
FIG_DIR = "figures";

% BASE_PREFIX = "1138_bus_isvd";
% BASE_PREFIX = "ex3_isvd";
% BASE_PREFIX = "plat1919_isvd";
% BASE_PREFIX = "kernel_stocks_1000_0.7071_isvd";
% BASE_PREFIX = "kernel_stocks_1000_1.0_isvd";
% BASE_PREFIX = "kernel_stocks_1000_2.2361_isvd";
% BASE_PREFIX = "bad_case1_1000_isvd";
% BASE_PREFIX = "bad_case2_1000_isvd";
% BASE_PREFIX = "bad_case3_1000_isvd";

SIZE = 110;
MAT_SIZE = 1000;
MAT_SIZE = 1138;
RESERVOIR_METHOD = "greedy";

% false -> sort primarily by sketch_size (\ell), then by window_size (W)
% true  -> sort by window_size (W)
SORT_BY_WINDOW_SIZE = false;

TARGET_RANK = 1;

SHOW_BAND = true;

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

if ~isfolder(FIG_DIR+"_jpg")
    mkdir(FIG_DIR+"_jpg");
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

    fprintf('  parsed prefix=%s size=%d W=%d ell=%d reservoir=%s seed=%d\n', ...
        parsed.prefix, parsed.size, parsed.window_size, parsed.sketch_size, ...
        parsed.reservoir, parsed.seed);

    if parsed.prefix ~= BASE_PREFIX
        fprintf('  rejected: prefix mismatch (wanted %s)\n', BASE_PREFIX);
        continue;
    end

    if parsed.reservoir ~= RESERVOIR_METHOD
        fprintf('  rejected: reservoir mismatch (wanted %s)\n', RESERVOIR_METHOD);
        continue;
    end

    if parsed.window_size + parsed.sketch_size ~= parsed.size
        fprintf('  rejected: W + ell = %d, but size = %d\n', ...
            parsed.window_size + parsed.sketch_size, parsed.size);
        continue;
    end

    key = sprintf('%d__%d', parsed.window_size, parsed.sketch_size);

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

for gi = 1:numel(group_keys)
    parts = split(string(group_keys{gi}), "__");
    group_window_size(gi) = str2double(parts(1));
    group_sketch_size(gi) = str2double(parts(2));
end

disp(group_keys)
keep_mask = group_sketch_size >= TARGET_RANK & group_window_size ~= 16 & group_window_size ~= 64;
group_keys = group_keys(keep_mask);
group_window_size = group_window_size(keep_mask);
group_sketch_size = group_sketch_size(keep_mask);

if SORT_BY_WINDOW_SIZE
    [~, order] = sort(group_window_size);
    group_keys = group_keys(order);
    group_window_size = group_window_size(order);
    group_sketch_size = group_sketch_size(order);
else
    [~, order] = sortrows([group_sketch_size(:), group_window_size(:)], [1 2]);
    group_keys = group_keys(order);
    group_window_size = group_window_size(order);
    group_sketch_size = group_sketch_size(order);
end

sorted_keys = [group_window_size(:), group_sketch_size(:)];

%% =========================
% Containers for later inspection in workspace
% ==========================
plot_results = struct();
plot_results.groups = groups;
plot_results.sorted_keys = sorted_keys;
plot_results.residual_fro_ell4 = {};
plot_results.residual_fro_ell16 = {};

used_any_residual_fro_ell4 = false;
used_any_residual_fro_ell16 = false;

%% =========================
% Create figure
% ==========================
fig_handle = figure('Position', [100 100 1600 700]);
setappdata(fig_handle, 'SelectedLine', gobjects(1));

ax_ell4 = subplot(1,2,1);
hold(ax_ell4, 'on');

ax_ell16 = subplot(1,2,2);
hold(ax_ell16, 'on');

%% =========================
% Main loop over (W, ell)
% ==========================
disp(group_keys)
for idx = 1:numel(group_keys)

    key = group_keys{idx};
    W = group_window_size(idx);
    ell = group_sketch_size(idx);

    seed_entries = groups(key);
    this_size = seed_entries{1}.parsed.size;

    seed_values = zeros(1, numel(seed_entries));
    for j = 1:numel(seed_entries)
        seed_values(j) = seed_entries{j}.seed;
    end
    [~, seed_order] = sort(seed_values);
    seed_entries = seed_entries(seed_order);

    residual_curves = {};
    seeds_used_residual = [];
    residual_curve_info = {};

    for j = 1:numel(seed_entries)

        seed = seed_entries{j}.seed;
        folder = seed_entries{j}.folder;
        exp_dir = fullfile(UNZIP_DIR, folder);

        [rcurve, residual_source_kind, residual_iters] = ...
            local_load_targetrank_wholespace_fro_curve(exp_dir, TARGET_RANK);

        if ~isempty(rcurve)
            rcurve = max(rcurve(:).', eps);

            residual_curves{end+1} = rcurve;
            seeds_used_residual(end+1) = seed;

            tmp = struct();
            tmp.seed = seed;
            tmp.folder = folder;
            tmp.exp_dir = exp_dir;
            tmp.curve = rcurve;
            tmp.iters = residual_iters;
            tmp.source_kind = residual_source_kind;

            residual_curve_info{end+1} = tmp;
        end
    end

    if ~isempty(residual_curves)

        residual_iter_lists = cell(1, numel(residual_curve_info));
        for ii = 1:numel(residual_curve_info)
            residual_iter_lists{ii} = residual_curve_info{ii}.iters;
        end

        [mean_curve, low_curve, high_curve, mean_endpoint, common_iters, aligned_log10] = ...
            local_aggregate_seed_curves(residual_curves, residual_iter_lists);

        if ~isempty(common_iters)
            x = local_load_x_from_window_info(seed_entries{1}.folder, UNZIP_DIR, common_iters);

            if isempty(x)
                fprintf('Residual (size=%d, W=%d, ell=%d): skipped plotting because window_info missing\n', ...
                    this_size, W, ell);
                continue;
            end

            label = sprintf('W=%d, ell=%d, end=%.3e, n=%d', ...
                W, ell, mean_endpoint, numel(residual_curves));

            line_info = struct();
            if ell == 4
                line_info.panel = "residual_fro_ell4";
            elseif ell == 16
                line_info.panel = "residual_fro_ell16";
            else
                line_info.panel = "residual_fro_other";
            end
            line_info.window_size = W;
            line_info.sketch_size = ell;
            line_info.W = W;
            line_info.ell = ell;
            line_info.mean_endpoint = mean_endpoint;
            line_info.nseeds = numel(residual_curves);
            line_info.seeds_used = seeds_used_residual;
            line_info.common_iters = common_iters;
            line_info.label = label;
            line_info.target_rank = TARGET_RANK;
            line_info.source_kind = residual_curve_info{1}.source_kind;

            if ell == 4
                h = semilogy(ax_ell4, x, mean_curve, '-o', ...
                    'LineWidth', 1.5, ...
                    'DisplayName', label, ...
                    'UserData', line_info, ...
                    'ButtonDownFcn', @local_line_click_callback, ...
                    'PickableParts', 'all');

                if SHOW_BAND && numel(residual_curves) > 1
                    p = fill(ax_ell4, [x, fliplr(x)], [low_curve, fliplr(high_curve)], h.Color, ...
                        'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                    set(p, 'HitTest', 'off', 'PickableParts', 'none');
                end

                used_any_residual_fro_ell4 = true;
            elseif ell == 16
                h = semilogy(ax_ell16, x, mean_curve, '-o', ...
                    'LineWidth', 1.5, ...
                    'DisplayName', label, ...
                    'UserData', line_info, ...
                    'ButtonDownFcn', @local_line_click_callback, ...
                    'PickableParts', 'all');

                if SHOW_BAND && numel(residual_curves) > 1
                    p = fill(ax_ell16, [x, fliplr(x)], [low_curve, fliplr(high_curve)], h.Color, ...
                        'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                    set(p, 'HitTest', 'off', 'PickableParts', 'none');
                end

                used_any_residual_fro_ell16 = true;
            else
                continue;
            end

            out = struct();
            out.window_size = W;
            out.sketch_size = ell;
            out.W = W;
            out.ell = ell;
            out.x = x;
            out.common_iters = common_iters;
            out.mean_curve = mean_curve;
            out.low_curve = low_curve;
            out.high_curve = high_curve;
            out.mean_endpoint = mean_endpoint;
            out.npts = numel(common_iters);
            out.nseeds = numel(residual_curves);
            out.seeds_used = seeds_used_residual;
            out.seed_curves = residual_curves;
            out.seed_curve_info = residual_curve_info;
            out.aligned_log10 = aligned_log10;
            out.source_kind = residual_curve_info{1}.source_kind;
            out.target_rank = TARGET_RANK;

            if ell == 4
                plot_results.residual_fro_ell4{end+1} = out;
            elseif ell == 16
                plot_results.residual_fro_ell16{end+1} = out;
            end

            fprintf('ResidualFro(top-%d) (size=%d, W=%d, ell=%d) seeds used: [%s], mean endpoint=%.6e\n', ...
                TARGET_RANK, this_size, W, ell, num2str(seeds_used_residual), mean_endpoint);
            fprintf('  common residual iters: [%s]\n', num2str(common_iters));
            fprintf('  usable residual curves: %d\n', numel(residual_curves));
        end
    end
end

%% =========================
% Axis styling
% ==========================
set(ax_ell4, 'YScale', 'log');
set(ax_ell16, 'YScale', 'log');

grid(ax_ell4, 'on');
ax_ell4.XMinorGrid = 'on';
ax_ell4.YMinorGrid = 'on';
ax_ell4.GridLineStyle = '--';
ax_ell4.GridAlpha = 0.5;
xlabel(ax_ell4, 'Rows processed');
ylabel(ax_ell4, sprintf('Whole-space residual Fro norm (top-%d)', TARGET_RANK));
title(ax_ell4, sprintf('Whole-space residual Fro: %s_random_uniform[*], ell=4\nvariable sizes, reservoir=%s, top-%d', ...
    BASE_PREFIX, RESERVOIR_METHOD, TARGET_RANK), 'Interpreter', 'none');

grid(ax_ell16, 'on');
ax_ell16.XMinorGrid = 'on';
ax_ell16.YMinorGrid = 'on';
ax_ell16.GridLineStyle = '--';
ax_ell16.GridAlpha = 0.5;
xlabel(ax_ell16, 'Rows processed');
ylabel(ax_ell16, sprintf('Whole-space residual Fro norm (top-%d)', TARGET_RANK));
title(ax_ell16, sprintf('Whole-space residual Fro: %s_random_uniform[*], ell=16\nvariable sizes, reservoir=%s, top-%d', ...
    BASE_PREFIX, RESERVOIR_METHOD, TARGET_RANK), 'Interpreter', 'none');

if used_any_residual_fro_ell4
    legend(ax_ell4, 'Location', 'best', 'FontSize', 8);
end
if used_any_residual_fro_ell16
    legend(ax_ell16, 'Location', 'best', 'FontSize', 8);
end

yl1 = ylim(ax_ell4);
yl2 = ylim(ax_ell16);
yl = [min([yl1(1), yl2(1)]), max([yl1(2), yl2(2)])];
ylim(ax_ell4, yl);
ylim(ax_ell16, yl);

%% =========================
% Save as MATLAB figure
% ==========================
fig_save_base = fullfile( ...
    FIG_DIR, ...
    sprintf('%s_random_uniform_allseeds_size_%d_residualfro_top%d_ell4_ell16_compare', ...
    BASE_PREFIX, SIZE, TARGET_RANK) ...
);

savefig(fig_handle, fig_save_base + ".fig");
fprintf('\nSaved MATLAB figure to: %s\n', fig_save_base + ".fig");

%% =========================
% Save as JPG with compression
% ==========================
fig_save_base = fullfile( ...
    FIG_DIR+"_jpg", ...
    sprintf('%s_random_uniform_allseeds_size_%d_residualfro_top%d_fix_sketch_compare', ...
    BASE_PREFIX, SIZE, TARGET_RANK) ...
);

jpg_path = fig_save_base + ".jpg";

tmp_png = fig_save_base + "_tmp.png";
exportgraphics(fig_handle, tmp_png, 'Resolution', 200);

img = imread(tmp_png);

jpeg_quality = 40;
imwrite(img, jpg_path, 'jpg', 'Quality', jpeg_quality);

delete(tmp_png);
fprintf('Saved JPEG figure to: %s (Quality=%d)\n', jpg_path, jpeg_quality);

%% =========================
% Helpful workspace notes
% ==========================
% Main variables left in workspace:
%   groups
%   sorted_keys
%   plot_results
%   fig_handle
%   ax_ell4
%   ax_ell16
%
% Example:
%   plot_results.residual_fro_ell4{1}
%   plot_results.residual_fro_ell16{1}

%% =========================
% Local functions
% ==========================
function parsed = local_parse_folder(folder_name)
    parsed = [];

    folder_name = char(folder_name);

    % Disk naming still uses "_ssize_" and "_k_";
    % internally these are mapped to:
    %   ssize -> window_size (W)
    %   k     -> sketch_size (ell)

    expr_seeded = ['^(?<prefix>.+)_random_uniform_(?<seed>\d+)' ...
                   '_size_(?<size>\d+)' ...
                   '_ssize_(?<window_size>\d+)' ...
                   '_k_(?<sketch_size>\d+)' ...
                   '_reservoir_(?<reservoir>.+)$'];

    m = regexp(folder_name, expr_seeded, 'names');

    if ~isempty(m)
        parsed = struct();
        parsed.prefix = string(m.prefix);
        parsed.seed = str2double(m.seed);
        parsed.size = str2double(m.size);
        parsed.window_size = str2double(m.window_size);
        parsed.sketch_size = str2double(m.sketch_size);
        parsed.W = parsed.window_size;
        parsed.ell = parsed.sketch_size;
        parsed.reservoir = string(m.reservoir);
        return;
    end

    expr_unseeded = ['^(?<prefix>.+)_random_uniform' ...
                     '_size_(?<size>\d+)' ...
                     '_ssize_(?<window_size>\d+)' ...
                     '_k_(?<sketch_size>\d+)' ...
                     '_reservoir_(?<reservoir>.+)$'];

    m = regexp(folder_name, expr_unseeded, 'names');

    if ~isempty(m)
        parsed = struct();
        parsed.prefix = string(m.prefix);
        parsed.seed = 1;
        parsed.size = str2double(m.size);
        parsed.window_size = str2double(m.window_size);
        parsed.sketch_size = str2double(m.sketch_size);
        parsed.W = parsed.window_size;
        parsed.ell = parsed.sketch_size;
        parsed.reservoir = string(m.reservoir);
        return;
    end
end

function out = local_load_txt(filename)
    txt = fileread(filename);
    raw = jsondecode(txt);

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

function [curve, source_kind, used_iters] = local_load_targetrank_wholespace_fro_curve(exp_dir, rank_limit)
    curve = [];
    source_kind = "";
    used_iters = [];

    residual_iters = local_list_consecutive_iterations(exp_dir, "reservoir_residuals_data");

    if isempty(residual_iters)
        return;
    end

    vals = [];

    for ii = 1:numel(residual_iters)
        j = residual_iters(ii);
        path = local_find_txt_file(exp_dir, "reservoir_residuals_data", j);

        if path == ""
            continue;
        end

        data = local_load_txt(path);

        if ~isfield(data, 'regular_residuals')
            continue;
        end

        r = double(data.regular_residuals(:));
        rr = min(rank_limit, numel(r));

        if rr <= 0
            continue;
        end

        vals(end+1) = norm(r(1:rr), 2); %#ok<AGROW>
        used_iters(end+1) = j; %#ok<AGROW>
    end

    if ~isempty(vals)
        curve = vals(:).';
        source_kind = sprintf('sqrt(sum_{i<=%d} regular_residuals(i)^2)', rank_limit);
    end
end

function [mean_curve, low_curve, high_curve, mean_endpoint, common_iters, arr_log10] = ...
    local_aggregate_seed_curves(curves, iter_lists)

    ncurves = numel(curves);

    if ncurves == 0
        mean_curve = [];
        low_curve = [];
        high_curve = [];
        mean_endpoint = [];
        common_iters = [];
        arr_log10 = [];
        return;
    end

    common_iters = iter_lists{1};

    for ii = 2:ncurves
        common_iters = intersect(common_iters, iter_lists{ii});
    end

    common_iters = sort(common_iters);

    if isempty(common_iters)
        mean_curve = [];
        low_curve = [];
        high_curve = [];
        mean_endpoint = [];
        arr_log10 = [];
        return;
    end

    arr_log10 = zeros(ncurves, numel(common_iters));

    for ii = 1:ncurves
        [tf, loc] = ismember(common_iters, iter_lists{ii});
        if ~all(tf)
            mean_curve = [];
            low_curve = [];
            high_curve = [];
            mean_endpoint = [];
            common_iters = [];
            arr_log10 = [];
            return;
        end
        vals = curves{ii}(loc);
        arr_log10(ii, :) = log10(vals);
    end

    mean_curve = 10.^mean(arr_log10, 1);
    low_curve = 10.^min(arr_log10, [], 1);
    high_curve = 10.^max(arr_log10, [], 1);
    mean_endpoint = 10.^mean(arr_log10(:, end));
end

function local_line_click_callback(lineH, ~)
    fig = ancestor(lineH, 'figure');

    oldH = getappdata(fig, 'SelectedLine');
    if ~isempty(oldH) && isgraphics(oldH)
        set(oldH, 'LineWidth', 1.5);
        set(oldH, 'MarkerSize', 6);
    end

    set(lineH, 'LineWidth', 3);
    set(lineH, 'MarkerSize', 9);
    setappdata(fig, 'SelectedLine', lineH);

    info = get(lineH, 'UserData');

    fprintf('\n=== Clicked line ===\n');
    if isfield(info, 'panel')
        fprintf('panel: %s\n', info.panel);
    end
    if isfield(info, 'window_size')
        fprintf('window_size (W): %d\n', info.window_size);
    elseif isfield(info, 'W')
        fprintf('window_size (W): %d\n', info.W);
    end
    if isfield(info, 'sketch_size')
        fprintf('sketch_size (ell): %d\n', info.sketch_size);
    elseif isfield(info, 'ell')
        fprintf('sketch_size (ell): %d\n', info.ell);
    end
    if isfield(info, 'mean_endpoint')
        fprintf('mean endpoint: %.6e\n', info.mean_endpoint);
    end
    if isfield(info, 'nseeds')
        fprintf('nseeds: %d\n', info.nseeds);
    end
    if isfield(info, 'seeds_used')
        fprintf('seeds used: [%s]\n', num2str(info.seeds_used));
    end
    if isfield(info, 'common_iters')
        fprintf('common iters: [%s]\n', num2str(info.common_iters));
    end
    if isfield(info, 'source_kind')
        fprintf('source: %s\n', info.source_kind);
    end
    if isfield(info, 'target_rank')
        fprintf('target rank: %d\n', info.target_rank);
    end
    if isfield(info, 'label')
        fprintf('label: %s\n', info.label);
    end
end

function x = local_load_x_from_window_info(folder_name, unzip_dir, iters)
    x = [];

    if isempty(iters)
        return;
    end

    exp_dir = fullfile(unzip_dir, folder_name);
    vals = nan(1, numel(iters));

    for ii = 1:numel(iters)
        j = iters(ii);
        path = local_find_txt_file(exp_dir, "window_info", j);

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