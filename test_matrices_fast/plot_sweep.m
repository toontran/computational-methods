clc;

%% =========================
% User settings
% ==========================
% ZIP_DIR = "D:\zips\sampled_zips";
% ZIP_DIR = "D:\zips\kernel_win_v_stream_zips";
% ZIP_DIR = "D:\zips\fix_win1_zips";

% UNZIP_DIR = "unzipped_win1";
% FIG_DIR = "figures_matlab";
% 
% ZIP_DIR = "D:\zips\fix_res_zips";
% UNZIP_DIR = "unzipped_fixres";
% FIG_DIR = "figures";

% ZIP_DIR = "D:\zips\small_sweep_new_zips";
% UNZIP_DIR = "unzipped_sweep_small_new";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_1000_1.0_isvd";

% ZIP_DIR = "D:\zips\large_sweep_new_zips";
% UNZIP_DIR = "unzipped_sweep_large_new";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_100000_1.0_isvd";

% ZIP_DIR = "D:\zips\large_bmw_zips";
% UNZIP_DIR = "unzipped_bmw";
% FIG_DIR = "figures";
% BASE_PREFIX = "bmw7st_1_isvd";

% ZIP_DIR = "D:\zips\kernel_0.2_zips";
% UNZIP_DIR = "unzipped_kernel_0.2";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_100000_0.2236_isvd";

% ZIP_DIR = "D:\zips\small_entropy_zips";
% UNZIP_DIR = "unzipped_small_entropy_zips";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_1000_0.2236_entropyscore_fast";


% ZIP_DIR = "D:\zips\small_isvd_zips";
% UNZIP_DIR = "unzipped_small_isvd_zips";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_1000_0.2236_isvd";

% ZIP_DIR = "D:\zips\small_ent_new_zips";
% UNZIP_DIR = "unzipped_small_ent_new_zips";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_1000_0.2236_entropyscore_fast";

ZIP_DIR = "D:\zips\small_ent_forget5_zips";
UNZIP_DIR = "unzipped_small_ent_forget5_zips";
FIG_DIR = "figures";
BASE_PREFIX = "kernel_stocks_1000_0.2236_entropyscore_forget";

% ZIP_DIR = "D:\zips\small_ent_forget2_zips";
% UNZIP_DIR = "unzipped_small_ent_forget2_zips";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_1000_0.2236_entropyscore_forget";

% ZIP_DIR = "D:\zips\small_ent_forget3_zips";
% UNZIP_DIR = "unzipped_small_ent_forget3_zips";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_1000_0.2236_entropyscore_forget";

% ZIP_DIR = "D:\zips\small_ent_expansion_zips";
% UNZIP_DIR = "unzipped_small_ent_expansion_zips";
% FIG_DIR = "figures";
% BASE_PREFIX = "kernel_stocks_1000_0.2236_entropyscore_expansion";

% ZIP_DIR = "D:\zips\1138_bus_zips";
% UNZIP_DIR = "unzipped_1138_bus_zips;";
% FIG_DIR = "figures";
% BASE_PREFIX = "1138_bus_entropyscore_forget";

ZIP_DIR = "D:\zips\output_sampled_zips";
UNZIP_DIR = "unzipped_output_sampled_zips;";
FIG_DIR = "figures";
BASE_PREFIX = "kernel_stocks_1000_0.2236_entropyscore_forget";

% ZIP_DIR = "D:\zips\output22_sampled_zips";
% UNZIP_DIR = "unzipped_output22_sampled_zips;";
% FIG_DIR = "figures";
% BASE_PREFIX = "1138_bus_entropyscore_forget";

% ZIP_DIR = "D:\zips\output22_sampled_zips";
% UNZIP_DIR = "unzipped_output22_sampled_zips;";
% FIG_DIR = "figures";
% BASE_PREFIX = "1138_bus_entropyscore_forget";

% ZIP_DIR = "D:\zips\1138_isvd_bus_zips";
% UNZIP_DIR = "unzipped_1138_isvd_bus_zips;";
% FIG_DIR = "figures";
% BASE_PREFIX = "1138_bus_isvd";

% ZIP_DIR = "D:\zips\1138_isvd_bus_zips";
% UNZIP_DIR = fullfile(SCRIPT_DIR, "unzipped_1138_isvd_bus_zips");
% FIG_DIR = fullfile(SCRIPT_DIR, "figures");
% BASE_PREFIX = "1138_bus_isvd";

% ZIP_DIR = "D:\zips\cfd_zips";
% UNZIP_DIR = "unzipped_cfd";
% FIG_DIR = "figures";
% BASE_PREFIX = "cfd2_isvd";

% BASE_PREFIX = "1138_bus_isvd";
% BASE_PREFIX = "ex3_isvd";
% BASE_PREFIX = "plat1919_isvd";
% BASE_PREFIX = "bad_case1_1000_isvd";
% BASE_PREFIX = "bad_case2_1000_isvd";
% BASE_PREFIX = "bad_case3_1000_isvd";
% BASE_PREFIX = "kernel_stocks_1000_0.7071_isvd";
% BASE_PREFIX = "kernel_stocks_1000_1.0_isvd";
% BASE_PREFIX = "kernel_stocks_1000_2.2361_isvd";

SIZE = 110;
MAT_SIZE = 1138;
RESERVOIR_METHOD = "greedy";
X_AXIS_MODE = "score_rank";   % "window" or "score_rank"
FIXED_SCORE_RANK = 16;    % used when X_AXIS_MODE == "window"
FIXED_WINDOW_SIZE = 10;  % used when X_AXIS_MODE == "score_rank"

% false -> sort primarily by sketch size, then by selected x-axis
% true  -> sort primarily by selected x-axis
SORT_BY_X_AXIS = false;

TARGET_RANK = 10;

SHOW_ERROR_BARS = true;

% "trace"    -> last-window relative trace error (current behavior)
% "residual" -> final residual endpoint using regular_residuals, like fix-sketch
METRIC_MODE = "trace";

% whether to normalize exact spectrum by sigma_1 for the spectrum subplot
NORMALIZE_SPECTRUM = true;

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

available_window_sizes = unique(sort(group_window_size(:)));
available_score_ranks = unique(sort(group_score_rank(~isnan(group_score_rank))));

if X_AXIS_MODE == "window"
    if isempty(FIXED_SCORE_RANK)
        error('Set FIXED_SCORE_RANK when X_AXIS_MODE is "window".');
    end
    keep_mask = group_sketch_size >= TARGET_RANK & ...
        group_window_size ~= 64 & ...
        ~isnan(group_score_rank) & group_score_rank == FIXED_SCORE_RANK;
    xAxisValues = group_window_size;
    xAxisLabel = 'Window size W';
    fixedAxisLabel = 'score rank';
    fixedAxisValue = FIXED_SCORE_RANK;
elseif X_AXIS_MODE == "score_rank"
    if isempty(FIXED_WINDOW_SIZE)
        error('Set FIXED_WINDOW_SIZE when X_AXIS_MODE is "score_rank".');
    end
    keep_mask = group_sketch_size >= TARGET_RANK & ...
        group_window_size == FIXED_WINDOW_SIZE & ...
        ~isnan(group_score_rank);
    xAxisValues = group_score_rank;
    xAxisLabel = 'Score rank';
    fixedAxisLabel = 'window size W';
    fixedAxisValue = FIXED_WINDOW_SIZE;
else
    error('Unknown X_AXIS_MODE: %s', X_AXIS_MODE);
end

group_keys = group_keys(keep_mask);
group_window_size = group_window_size(keep_mask);
group_sketch_size = group_sketch_size(keep_mask);
group_score_rank = group_score_rank(keep_mask);
xAxisValues = xAxisValues(keep_mask);

if isempty(group_keys)
    fprintf('Available window sizes: [%s]\n', num2str(available_window_sizes.'));
    fprintf('Available score ranks: [%s]\n', num2str(available_score_ranks.'));
    fprintf('Available (W, ell, sr) triples:\n');
    disp([group_window_size(:), group_sketch_size(:), group_score_rank(:)]);
    error('No experiment groups matched X_AXIS_MODE=%s with fixed %s=%d.', ...
        X_AXIS_MODE, fixedAxisLabel, fixedAxisValue);
end

if SORT_BY_X_AXIS
    [~, order] = sort(xAxisValues);
    group_keys = group_keys(order);
    group_window_size = group_window_size(order);
    group_sketch_size = group_sketch_size(order);
    group_score_rank = group_score_rank(order);
    xAxisValues = xAxisValues(order);
else
    [~, order] = sortrows([group_sketch_size(:), xAxisValues(:)], [1 2]);
    group_keys = group_keys(order);
    group_window_size = group_window_size(order);
    group_sketch_size = group_sketch_size(order);
    group_score_rank = group_score_rank(order);
    xAxisValues = xAxisValues(order);
end

sorted_keys = [xAxisValues(:), group_sketch_size(:), group_score_rank(:)];

%% =========================
% Load one reference exact spectrum
% Assumption: all trajectories share the same exact spectrum
% ==========================
[spectrum_exact_raw, spectrum_exact_plot, spectrum_source_folder] = ...
    local_load_reference_spectrum(groups, group_keys, UNZIP_DIR, NORMALIZE_SPECTRUM);

sigma_k = NaN;
sigma_k1 = NaN;
spectral_gap = NaN;

if ~isempty(spectrum_exact_plot) && numel(spectrum_exact_plot) >= TARGET_RANK
    sigma_k = spectrum_exact_plot(TARGET_RANK);
    if numel(spectrum_exact_plot) >= TARGET_RANK + 1
        sigma_k1 = spectrum_exact_plot(TARGET_RANK + 1);
        spectral_gap = sigma_k - sigma_k1;
    end
end

%% =========================
% Collect last summary per (W, ell)
% ==========================
plot_results = struct();
plot_results.groups = groups;
plot_results.sorted_keys = sorted_keys;
plot_results.metric_mode = METRIC_MODE;
plot_results.last_window_summary = {};
plot_results.spectrum_exact_raw = spectrum_exact_raw;
plot_results.spectrum_exact_plot = spectrum_exact_plot;
plot_results.spectrum_source_folder = spectrum_source_folder;
plot_results.target_rank = TARGET_RANK;
plot_results.sigma_k = sigma_k;
plot_results.sigma_k1 = sigma_k1;
plot_results.spectral_gap = spectral_gap;

xAxisPlot = [];
yEll = [];
zMean = [];
zLow = [];
zHigh = [];
nSeedsUsed = [];
seedLists = {};
folderLists = {};

for idx = 1:numel(group_keys)

    key = group_keys{idx};
    W = group_window_size(idx);
    ell = group_sketch_size(idx);
    sr = group_score_rank(idx);

    seed_entries = groups(key);
    this_size = seed_entries{1}.parsed.size;

    seed_values = zeros(1, numel(seed_entries));
    for j = 1:numel(seed_entries)
        seed_values(j) = seed_entries{j}.seed;
    end
    [~, seed_order] = sort(seed_values);
    seed_entries = seed_entries(seed_order);

    endpoint_vals = [];
    seeds_used = [];
    folders_used = strings(1,0);
    endpoint_iters = [];

    for j = 1:numel(seed_entries)
        seed = seed_entries{j}.seed;
        folder = seed_entries{j}.folder;
        exp_dir = fullfile(UNZIP_DIR, folder);

        if METRIC_MODE == "trace"
            [curve_vals, curve_iters] = local_load_trace_error_curve(exp_dir, TARGET_RANK);
        elseif METRIC_MODE == "residual"
            [curve_vals, curve_iters] = local_load_residual_endpoint_curve(exp_dir, TARGET_RANK);
        else
            error('Unknown METRIC_MODE: %s', METRIC_MODE);
        end

        if ~isempty(curve_vals) && ~isempty(curve_iters)
            endpoint_vals(end+1) = max(curve_vals(end), eps); %#ok<AGROW>
            endpoint_iters(end+1) = curve_iters(end); %#ok<AGROW>
            seeds_used(end+1) = seed; %#ok<AGROW>
            folders_used(end+1) = folder;
        end
    end

    if isempty(endpoint_vals)
        fprintf('No usable endpoint for (W=%d, ell=%d)\n', W, ell);
        continue;
    end

    log_vals = log10(endpoint_vals);
    mean_endpoint = 10.^mean(log_vals);
    low_endpoint  = 10.^min(log_vals);
    high_endpoint = 10.^max(log_vals);

    xAxisPlot(end+1) = xAxisValues(idx); %#ok<AGROW>
    yEll(end+1) = ell; %#ok<AGROW>
    zMean(end+1) = mean_endpoint; %#ok<AGROW>
    zLow(end+1) = low_endpoint; %#ok<AGROW>
    zHigh(end+1) = high_endpoint; %#ok<AGROW>
    nSeedsUsed(end+1) = numel(endpoint_vals); %#ok<AGROW>
    seedLists{end+1} = seeds_used; %#ok<AGROW>
    folderLists{end+1} = folders_used; %#ok<AGROW>

    out = struct();
    out.window_size = W;
    out.sketch_size = ell;
    out.W = W;
    out.ell = ell;
    out.score_rank = sr;
    out.metric_mode = METRIC_MODE;
    out.endpoint_vals = endpoint_vals;
    out.endpoint_iters = endpoint_iters;
    out.mean_endpoint = mean_endpoint;
    out.low_endpoint = low_endpoint;
    out.high_endpoint = high_endpoint;
    out.nseeds = numel(endpoint_vals);
    out.seeds_used = seeds_used;
    out.folders_used = folders_used;

    plot_results.last_window_summary{end+1} = out;

    fprintf('LastWindowSummary (%s, size=%d, W=%d, ell=%d, sr=%s): mean=%.6e, low=%.6e, high=%.6e, n=%d\n', ...
        METRIC_MODE, this_size, W, ell, local_score_rank_label(sr), mean_endpoint, low_endpoint, high_endpoint, numel(endpoint_vals));
    fprintf('  seeds used: [%s]\n', num2str(seeds_used));
    fprintf('  last iters: [%s]\n', num2str(endpoint_iters));
end

%% =========================
% Figure 1: 3D endpoint summary
% ==========================
fig_handle = figure('Position', [100 100 900 700]);
ax3d = axes(fig_handle);
hold(ax3d, 'on');

hs = scatter3(ax3d, xAxisPlot, yEll, zMean, 80, zMean, 'filled');
set(hs, 'MarkerEdgeColor', 'k');

if SHOW_ERROR_BARS
    for i = 1:numel(xAxisPlot)
        plot3(ax3d, [xAxisPlot(i) xAxisPlot(i)], [yEll(i) yEll(i)], [zLow(i) zHigh(i)], ...
            'k-', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        plot3(ax3d, [xAxisPlot(i)-0.8 xAxisPlot(i)+0.8], [yEll(i) yEll(i)], [zLow(i) zLow(i)], ...
            'k-', 'LineWidth', 1.0, 'HandleVisibility', 'off');
        plot3(ax3d, [xAxisPlot(i)-0.8 xAxisPlot(i)+0.8], [yEll(i) yEll(i)], [zHigh(i) zHigh(i)], ...
            'k-', 'LineWidth', 1.0, 'HandleVisibility', 'off');
    end
end

set(ax3d, 'ZScale', 'log');
grid(ax3d, 'on');
ax3d.XMinorGrid = 'on';
ax3d.YMinorGrid = 'on';
ax3d.ZMinorGrid = 'on';
ax3d.GridLineStyle = '--';
ax3d.GridAlpha = 0.5;

xlabel(ax3d, xAxisLabel);
ylabel(ax3d, 'Sketch size ell');

if METRIC_MODE == "trace"
    zlabel(ax3d, sprintf('Last-window relative trace error (top-%d)', TARGET_RANK));
    metric_title = sprintf('Last-window trace error summary');
else
    zlabel(ax3d, sprintf('Final residual endpoint (top-%d)', TARGET_RANK));
    metric_title = sprintf('Final residual summary');
end

if isnan(spectral_gap)
    gap_str = sprintf('\\sigma_{%d}-\\sigma_{%d} = N/A', TARGET_RANK, TARGET_RANK+1);
else
    gap_str = sprintf('\\sigma_{%d}-\\sigma_{%d} = %.2e', TARGET_RANK, TARGET_RANK+1, spectral_gap);
end

title(ax3d, sprintf(['%s: %s\n' ...
    'reservoir=%s, size=%d, fixed %s=%d, %s'], ...
    metric_title, BASE_PREFIX, ...
    RESERVOIR_METHOD, SIZE, fixedAxisLabel, fixedAxisValue, gap_str), ...
    'Interpreter', 'tex');

view(ax3d, 45, 28);
colorbar(ax3d);

for i = 1:numel(xAxisPlot)
    txt = sprintf('(%d,%d), n=%d', xAxisPlot(i), yEll(i), nSeedsUsed(i));
    text(ax3d, xAxisPlot(i), yEll(i), zMean(i), ['  ' txt], 'FontSize', 8);
end

% x0 = 20000 * ones(100);
% y = linspace(1,10,100);
% z = 1./(1+y);
% plot3(ax3d, x0,y,z);


%% =========================
% Figure 2: exact spectrum
% ==========================
fig_spec = figure('Position', [150 150 900 700]);
axspec = axes(fig_spec);
hold(axspec, 'on');

if ~isempty(spectrum_exact_plot)
    semilogy(axspec, 1:numel(spectrum_exact_plot), max(spectrum_exact_plot(:).', eps), '-o', ...
        'LineWidth', 1.5, 'MarkerSize', 4);

    if TARGET_RANK <= numel(spectrum_exact_plot)
        xline(axspec, TARGET_RANK, '--', sprintf('k=%d', TARGET_RANK), ...
            'LabelOrientation', 'horizontal', 'HandleVisibility', 'off');
    end

    if TARGET_RANK + 1 <= numel(spectrum_exact_plot)
        xline(axspec, TARGET_RANK+1, '--', sprintf('k+1=%d', TARGET_RANK+1), ...
            'LabelOrientation', 'horizontal', 'HandleVisibility', 'off');
    end

    if NORMALIZE_SPECTRUM
        ylabel(axspec, 'Normalized exact singular values');
    else
        ylabel(axspec, 'Exact singular values');
    end

    yscale(axspec, 'log');
    xlabel(axspec, 'Singular index');
    grid(axspec, 'on');
    axspec.XMinorGrid = 'on';
    axspec.YMinorGrid = 'on';
    axspec.GridLineStyle = '--';
    axspec.GridAlpha = 0.5;

    if isnan(spectral_gap)
        spec_title = sprintf('Exact spectrum (source: %s)', spectrum_source_folder);
    else
        spec_title = sprintf('Exact spectrum, gap \\sigma_{%d}-\\sigma_{%d} = %.6e', ...
            TARGET_RANK, TARGET_RANK+1, spectral_gap);
    end

    title(axspec, spec_title, 'Interpreter', 'tex');
else
    axis(axspec, 'off');
    text(axspec, 0.05, 0.5, 'No exact spectrum found', 'FontSize', 12);
end

%% =========================
% Save 3D figure as MATLAB figure
% ==========================
fig_metric_tag = char(METRIC_MODE);

fig_save_base = fullfile( ...
    FIG_DIR, ...
    sprintf('%s%s_random_uniform_allseeds_size_%d_%s_3d', ...
    BASE_PREFIX, local_filter_file_suffix(X_AXIS_MODE, FIXED_WINDOW_SIZE, FIXED_SCORE_RANK), SIZE, fig_metric_tag) ...
);

savefig(fig_handle, fig_save_base + ".fig");
fprintf('\nSaved MATLAB figure to: %s\n', fig_save_base + ".fig");

%% =========================
% Save spectrum figure as MATLAB figure
% ==========================
fig_spec_base = fullfile( ...
    FIG_DIR, ...
    sprintf('%s%s_random_uniform_allseeds_size_%d_spectrum', ...
    BASE_PREFIX, local_filter_file_suffix(X_AXIS_MODE, FIXED_WINDOW_SIZE, FIXED_SCORE_RANK), SIZE) ...
);

savefig(fig_spec, fig_spec_base + ".fig");
fprintf('Saved MATLAB spectrum figure to: %s\n', fig_spec_base + ".fig");

%% =========================
% Save 3D figure as JPG with compression
% ==========================
fig_save_base_jpg = fullfile( ...
    FIG_DIR+"_jpg", ...
    sprintf('%s%s_random_uniform_allseeds_size_%d_%s_3d', ...
    BASE_PREFIX, local_filter_file_suffix(X_AXIS_MODE, FIXED_WINDOW_SIZE, FIXED_SCORE_RANK), SIZE, fig_metric_tag) ...
);

jpg_path = fig_save_base_jpg + ".jpg";

tmp_png = fig_save_base_jpg + "_tmp.png";
exportgraphics(fig_handle, tmp_png, 'Resolution', 200);

img = imread(tmp_png);
jpeg_quality = 40;
imwrite(img, jpg_path, 'jpg', 'Quality', jpeg_quality);

delete(tmp_png);

fprintf('Saved JPEG figure to: %s (Quality=%d)\n', jpg_path, jpeg_quality);

%% =========================
% Save spectrum figure as JPG with compression
% ==========================
fig_spec_base_jpg = fullfile( ...
    FIG_DIR+"_jpg", ...
    sprintf('%s%s_random_uniform_allseeds_size_%d_spectrum', ...
    BASE_PREFIX, local_filter_file_suffix(X_AXIS_MODE, FIXED_WINDOW_SIZE, FIXED_SCORE_RANK), SIZE) ...
);

jpg_spec_path = fig_spec_base_jpg + ".jpg";

tmp_spec_png = fig_spec_base_jpg + "_tmp.png";
print(fig_spec, tmp_spec_png, '-dpng', '-r200');

img_spec = imread(tmp_spec_png);
imwrite(img_spec, jpg_spec_path, 'jpg', 'Quality', jpeg_quality);

delete(tmp_spec_png);

fprintf('Saved JPEG spectrum figure to: %s (Quality=%d)\n', jpg_spec_path, jpeg_quality);

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
        parsed.W = parsed.window_size;
        parsed.ell = parsed.sketch_size;
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
        parsed.W = parsed.window_size;
        parsed.ell = parsed.sketch_size;
        parsed.score_rank = local_parse_score_rank(folder_name);
        parsed.reservoir = string(m.reservoir);
        return;
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
        out = sprintf(',sr=%d', score_rank);
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

function out = local_filter_file_suffix(axis_mode, fixed_window_size, fixed_score_rank)
    if axis_mode == "window"
        out = sprintf('_sr_%d', fixed_score_rank);
    elseif axis_mode == "score_rank"
        out = sprintf('_W_%d', fixed_window_size);
    else
        out = "";
    end
end

function out = local_filter_title_suffix(axis_mode, fixed_window_size, fixed_score_rank)
    if axis_mode == "window"
        out = sprintf(', fixed sr=%d', fixed_score_rank);
    elseif axis_mode == "score_rank"
        out = sprintf(', fixed W=%d', fixed_window_size);
    else
        out = "";
    end
end

function out = local_sortable_score_rank(score_rank)
    out = score_rank;
    out(isnan(out)) = -1;
end

function out = local_load_txt(filename)
    try
        txt = fileread(filename);
        raw = jsondecode(txt);
    catch
        fprintf('Skipping invalid JSON file: %s\n', filename);
        out = struct();   % return empty
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

    % Load S_exact once, from iteration 0 if possible, otherwise from the
    % first spectrum_data_* file that still contains it.
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
            j = spectrum_iters(ii);
            path = local_find_txt_file(exp_dir, "spectrum_data", j);
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

    if isempty(S_exact)
        return;
    end

    S_cells = {};
    for ii = 1:numel(spectrum_iters)
        j = spectrum_iters(ii);
        path = local_find_txt_file(exp_dir, "spectrum_data", j);

        if path == ""
            continue;
        end

        data = local_load_txt(path);

        % Now only S is required per iteration
        if ~isfield(data, 'S')
            continue;
        end

        S = double(data.S(:));
        S_cells{end+1} = S; %#ok<AGROW>
        used_iters(end+1) = j; %#ok<AGROW>
    end

    if isempty(S_cells)
        used_iters = [];
        return;
    end

    if S_exact(1) == 0
        return;
    end

    S_exact_norm = S_exact ./ S_exact(1);

    lengths = zeros(1, numel(S_cells));
    for ii = 1:numel(S_cells)
        lengths(ii) = numel(S_cells{ii});
    end

    min_rank = min([min(lengths), numel(S_exact_norm), rank_limit]);
    if min_rank <= 0
        return;
    end

    denom = sum(S_exact_norm(1:min_rank));
    if denom == 0
        return;
    end

    vals = zeros(1, numel(S_cells));
    for ii = 1:numel(S_cells)
        tr_S = sum(S_cells{ii}(1:min_rank));
        vals(ii) = abs(denom - tr_S) / abs(denom);
    end

    curve = vals(:).';
end

function [curve, used_iters] = local_load_residual_endpoint_curve(exp_dir, rank_limit)
    curve = [];
    used_iters = [];

    residual_sources = { ...
        struct('prefix', "reservoir_residuals_data", 'field', 'regular_residuals'), ...
        struct('prefix', "residuals_sym_psd_data", 'field', 'approx_residuals'), ...
        struct('prefix', "residuals_sym_psd_data_truncated", 'field', 'approx_residuals'), ...
        struct('prefix', "residuals_sym_psd_data_truncated_Rayleigh", 'field', 'approx_residuals') ...
    };

    vals = [];
    source_idx = [];

    for si = 1:numel(residual_sources)
        residual_iters = local_list_consecutive_iterations(exp_dir, residual_sources{si}.prefix);
        if isempty(residual_iters)
            continue;
        end

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
            used_iters(end+1) = j; %#ok<AGROW>
            source_idx(end+1) = si; %#ok<AGROW>
        end

        if ~isempty(vals)
            break;
        end
    end

    if ~isempty(vals)
        [used_iters, order] = sort(used_iters);
        vals = vals(order);
        source_idx = source_idx(order);

        % Keep one consistent residual source per experiment.
        keep_mask = source_idx == source_idx(1);
        curve = vals(keep_mask);
        used_iters = used_iters(keep_mask);
        curve = curve(:).';
    end
end

function [S_exact_raw, S_exact_plot, source_folder] = ...
    local_load_reference_spectrum(groups, group_keys, unzip_dir, normalize_spectrum)

    S_exact_raw = [];
    S_exact_plot = [];
    source_folder = "";

    for gi = 1:numel(group_keys)
        key = group_keys{gi};
        seed_entries = groups(key);

        for sj = 1:numel(seed_entries)
            folder = seed_entries{sj}.folder;
            exp_dir = fullfile(unzip_dir, folder);

            spectrum_iters = local_list_consecutive_iterations(exp_dir, "spectrum_data");
            if isempty(spectrum_iters)
                continue;
            end

            path = local_find_txt_file(exp_dir, "spectrum_data", spectrum_iters(1));
            if path == ""
                continue;
            end

            data = local_load_txt(path);
            if ~isfield(data, 'S_exact')
                continue;
            end

            S_exact_raw = double(data.S_exact(:));

            if isempty(S_exact_raw)
                continue;
            end

            if normalize_spectrum
                if S_exact_raw(1) == 0
                    S_exact_plot = S_exact_raw;
                else
                    S_exact_plot = S_exact_raw ./ S_exact_raw(1);
                end
            else
                S_exact_plot = S_exact_raw;
            end

            source_folder = folder;
            return;
        end
    end
end
