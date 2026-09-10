% Parameters
Ninit=1; % # of first initial image
Nref = 2; % # of last initial image
NB_DEF_TRAIN = 160; % # of first image for evaluation
NB_DEF_MAX_PER_IMG = 161; % # of last deformed image for evaluation

radius = 4;
spacing = 0;
cutoff_diffnorm = 0.0001;
cutoff_iteration = 100;
total_threads = 1;   % single seed -> no "all N seeds must converge" AND-condition
enabled_stepanalysis = false;
subsettrunc = true;
pos_parent = [100 100 800 600];

% --- Headless seed placement -------------------------------------------
% With enabled_stepanalysis = false, ncorr_alg_seedanalysis fails the
% WHOLE seeding step if even one of the pos_seed points fails to converge
% (see ncorr_alg_seedanalysis.m: "if any seed placements fail then fail
% the whole analysis"). With total_threads = 4 that means all 4 blind
% corner guesses had to succeed simultaneously - if even one landed on
% weak local texture, everything failed. Using a single seed removes that
% AND-condition, and trying a short list of fallback candidates (instead
% of only the center) makes this robust across all 199 reference images,
% whose speckle patterns will vary in where the strongest texture is.
seed_candidates = [64 64; 32 32; 96 96; 96 32; 32 96];
%seed_candidates = [64 64];
num_region = 0;

for i = Ninit:Nref
    for j = NB_DEF_TRAIN:NB_DEF_MAX_PER_IMG
        ref = sprintf("Ref_MO_128X128/Ref_image_%d.bmp", i);
        def = sprintf("Def_1px/GeneratedBoukhtache1px/Ref_image_%d_def_%03d.bmp", i, j);

        % Reference image
        img1 = ncorr_class_img();
        Iref = imread(ref);
        data_ref.img = Iref;
        data_ref.name = sprintf("Ref_image_%d.bmp", i);
        data_ref.path = "Ref_MO_128X128";
        img1.set_img('load', data_ref);

        % Deformed image
        img2 = ncorr_class_img();
        Idef = imread(def);
        data_def.img = Idef;
        data_def.name = sprintf("Ref_image_%d_def_%03d.bmp", i, j);
        data_def.path = "Def_1px/GeneratedBoukhtache1px";
        img2.set_img('load', data_def);

        % ROI
        roi = ncorr_class_roi();
        data_roi.mask = true(128, 128);
        data_roi.cutoff = 0;
        roi.set_roi('load', data_roi);

        % Package
        clear imgs
        imgs(1).imginfo = img1;
        imgs(1).roi = roi;
        imgs(2).imginfo = img2;
        imgs(2).roi = roi;

        % --- Compute seeds headlessly (no GUI), trying fallback candidates ---
        outstate_seeds = out.failed;
        for c = 1:size(seed_candidates, 1)
            pos_seed = seed_candidates(c, :);

            [seedinfo, convergence, outstate_seeds] = ncorr_alg_seedanalysis( ...
                img1, ...
                img2, ...
                roi, ...
                num_region, ...
                pos_seed, ...
                radius, ...
                cutoff_diffnorm, ...
                cutoff_iteration, ...
                enabled_stepanalysis, ...
                subsettrunc, ...
                1, ...   % num_img
                1);      % total_imgs

            if outstate_seeds == out.success
                break;
            end
        end

        if outstate_seeds ~= out.success
            fprintf("Seeding failed for i=%d j=%d on all %d candidates, skipping\n", i, j, size(seed_candidates, 1));
            continue;
        end

        params_init = seedinfo;  % non-empty -> ncorr_alg_dicanalysis skips the GUI

        % --- Run Ncorr ---------------------------------------------------
        [displacements, rois_dic, seedinfo_out, outstate] = ncorr_alg_dicanalysis( ...
            imgs, ...
            radius, ...
            spacing, ...
            cutoff_diffnorm, ...
            cutoff_iteration, ...
            total_threads, ...
            enabled_stepanalysis, ...
            subsettrunc, ...
            1, ...              % num_img
            1, ...              % total_imgs
            pos_parent, ...
            params_init);

        if outstate == out.success
            U = displacements.plot_u;
            V = displacements.plot_v;
        else
            fprintf("Ncorr failed for i=%d j=%d\n", i, j);
            continue;
        end
        
        % Save
        save(sprintf("Def_1px/GeneratedBoukhtache1px/Ncorr_dic/U_%d_def_%03d.mat", i, j), "U");
        save(sprintf("Def_1px/GeneratedBoukhtache1px/Ncorr_dic/V_%d_def_%03d.mat", i, j), "V");
    end
end