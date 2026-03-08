clear all;
% csf_model = CSF_castleCSF();

csf_model = CSF_castleCSF_TransLimited();
fit_data = load('E:\Matlab_codes\csf_datasets\model_fitting\fitted_models\CSF_MLP_CSF_stlae_lms_d_smooth_34_castleCSF_TransLimited/castle-csf-temporal-limit_all_2025-11-11_22-00.mat');
csf_model.par = CSF_base.update_struct( fit_data.fitted_struct, csf_model.par );
csf_model = csf_model.set_pars(csf_model.get_pars());
% Parameters
t_frequencies = [10, 20, 30, 45];  % Hz
luminances = [50, 5];               % nits
area = pi * 2^2;                    % Area of disk with radius 2

% LMS delta vectors for each channel
lms_ach = [0.68455967, 0.29620126, 0.01923907];  % Achromatic
lms_rg  = [0.5, -0.5, 0];                         % Red-Green
lms_yv  = [0, 0, 1];                              % Yellow-Violet

% Target contrasts
contrast_ach = 0.9;
contrast_rg  = 0.14;
contrast_yv  = 0.92;

% Target sensitivities (1 / contrast)
target_sens_ach = 1 / contrast_ach;
target_sens_rg  = 1 / contrast_rg;
target_sens_yv  = 1 / contrast_yv;

% Spatial frequency search range (4 to 64 cpd, step 0.01)
s_freqs = (4:0.01:64)';

% Channel definitions
channels    = {'Achromatic', 'Red-Green', 'Yellow-Violet'};
lms_deltas  = {lms_ach, lms_rg, lms_yv};
target_sens = [target_sens_ach, target_sens_rg, target_sens_yv];

% Results storage
results = {};
row = 1;

for c = 1:3
    for l = 1:length(luminances)
        lum = luminances(l);
        for t = 1:length(t_frequencies)
            tf = t_frequencies(t);

            % Build CSF parameters
            csf_pars = struct(...
                's_frequency',  s_freqs, ...
                't_frequency',  tf, ...
                'orientation',  0, ...
                'luminance',    lum, ...
                'area',         area, ...
                'eccentricity', 0, ...
                'lms_delta',    lms_deltas{c});

            % Compute sensitivities
            sensitivities = csf_model.sensitivity(csf_pars);

            % Find spatial frequency closest to target_sens in log scale
            log_diff = abs(log10(sensitivities) - log10(target_sens(c)));
            [min_diff, idx] = min(log_diff);

            sf_threshold    = s_freqs(idx);
            sens_at_thresh  = sensitivities(idx);

            % Reject if closest sensitivity deviates more than 5% from target
            if abs(sens_at_thresh - target_sens(c)) / target_sens(c) > 0.05
                sf_threshold   = NaN;
                sens_at_thresh = NaN;
            end

            results{row} = {channels{c}, tf, lum, sf_threshold, sens_at_thresh};
            row = row + 1;
        end
    end
end

% Print summary table
fprintf('\n=== SUMMARY TABLE ===\n');
fprintf('%-15s | %-10s | %-12s | %-22s | %-25s\n', ...
    'Channel', 'TF (Hz)', 'Lum (nits)', 'Max Detectable SF (cpd)', 'Sensitivity at threshold');
fprintf('%s\n', repmat('-', 1, 92));

for r = 1:length(results)
    ch  = results{r}{1};
    tf  = results{r}{2};
    lum = results{r}{3};
    sf  = results{r}{4};
    sen = results{r}{5};

    if isnan(sf)
        fprintf('%-15s | %-10d | %-12.0f | %-22s | %-25s\n', ch, tf, lum, 'NaN', 'NaN');
    else
        fprintf('%-15s | %-10d | %-12.0f | %-22.2f | %-25.4f\n', ch, tf, lum, sf, sen);
    end
end