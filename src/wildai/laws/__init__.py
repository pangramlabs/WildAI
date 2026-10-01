"""Scaling laws for wild AI text: the paper's law, the eleven published laws it is benchmarked against, fitting, scoring
and every law-derived number of the paper's tables and figures.

Regenerate ``results/laws/`` (about 40 CPU-minutes, spread over every core; under a minute on 128 cores)::

    python -m wildai.laws.run [--targets c4 fw22 ...] [--workers N] [--reuse-fits]

Our law (``ours_logshare_11``), in n = N/1e8, d = D_H/1e9, r = D_AI/D_H and t = D_H/(20 N)::

    L = E + A n^-alpha + B [d (1 + eta R* (1 - exp(-r / R*)))]^-beta (1 + gamma t^u n^v [log(1 + r) - r / (1 + r)]),   R* = K t^rho

Modules
    data              Study (models.csv + losses.csv), Run, Observations (runs on one target paired with their controls), Coordinates
    forms             Law (named, bounded coefficients; vectorised complex-step-safe ``predict``), Parameter, FittedLaw
    ours              Ours / OursForm: the paper's law and the one-ingredient variants of the law-form ablation (ABLATION_FORMS)
    comparators       Chinchilla, Muennighoff, CD, Lovelace, ATLAS, He, Hamidieh, Shukor (additive, joint), Sedova, Jain
    catalog           benchmark_laws() in table order, PAPER_LAW
    fit               the paired Huber objective, the deterministic multi-start (fit, starts_for, solve, select), LawFit
    bank              fit_cases: many (law, target, fold) cases on a process pool; FOLDS (all, without_<size>)
    score             held-out predictions and paired/absolute RMSEs (Predictions, Score)
    ceg               cost of not filtering, AI-token value, optimal AI share, ReferenceMix (CEG against a mix)
    records           output records and CSV/JSON writers
    analysis/         benchmark, stability (leave one size out), ablation, significance, placement, filtering, ceg_ladder,
                      recommendations, masking
    run               the CLI

Output files (``results/laws/``; RMSEs and changes in natural-log units, losses in bits per byte, never scaled by 1,000;
"held-out" = the 74 runs at 477M and 973M, fits use the 726 runs at 19.9M to 268M; runs by their public names)

fits.json               [FitRecord]: law, label, target, fold ("all" | "without_<size>"), k, runs, starts, objective (Huber
                        cost), converged, active_bounds, parameters (fitted values, logs of positive coefficients), coefficients
                        (the paper's symbols, e.g. E, A, alpha, B, beta, eta, K, rho, gamma, u, v). Benchmark laws on every
                        target (fold all); our law and the joint law of Shukor et al. on C4 for every fold; the law-form
                        ablation variants on C4 for every fold; the additive-harm variant on every target.
scores.csv              law, target, fold, depth (empty: both held-out sizes), cutoff (empty: every r; 1.0: r < 1), runs,
                        ai_runs, human_runs, paired_rmse (the paper's paired error, AI runs), paired_human_rmse (fresh-human
                        runs), absolute_ai_rmse, absolute_all_rmse.
predictions.csv         law, target, fold, name, split (fit | held_out), control, arm, depth, ratio, human_tpp, observed_bpb,
                        predicted_bpb, observed_change, predicted_change (log L/L_control). Fold "all": the held-out runs of
                        every benchmark law and target, and our law's fitted runs too; folds "without_<size>": the left-out
                        size's runs (C4; our law and Shukor joint).
stability.csv           law, target, fold, runs, objective, active_bounds, left_out_ai_runs, left_out_paired_rmse,
                        cost_of_not_filtering_8b, ai_token_value_8b (8B parameters, 20 TPP_h, the 2026 mix).
ablation.csv            name, group (benefit | harm | both), description, k, objective, active_bounds, held_out_ai_runs,
                        held_out_paired_rmse (the table's column), held_out_paired_rmse_477m / _973m / _low (r < 1),
                        leave_one_size_out_ai_runs, leave_one_size_out_paired_rmse, _low, _high (r >= 1).
significance.json       [SignificanceRecord] per human-text target and cut ("all" | "r<1"): runs, groups, rival
                        (shukor_joint), point {law: paired RMSE}, interval {law: 95 % bootstrap interval}, gap (rival minus
                        ours), gap_interval, p_ours_lower {law: share of draws}, p_ours_best, leave_one_group_out_gap,
                        groups_ours_lower.
harm_placement.json     [PlacementRecord] per target: ours, additive (held-out paired RMSE of the harm added to the loss),
                        ours_973m, additive_973m, gap, gap_interval, p_ours_lower, cost_of_not_filtering_8b {law: cost}.
filtering.csv           law, target, group, depth, n_params, human_tpp, ratio, observed_change, predicted_change: removing
                        the AI documents of the 2026 mix (log L_filtered / L_mix), measured and predicted.
filtering_summary.csv   law, target, pairs, paired_rmse, direction_right, filtering_helps, predicted_harm_where_it_helps.
ceg_ladder.csv          target, name, control, split, arm, depth, n_params, human_tokens, ai_tokens, ratio, ai_share,
                        observed_bpb, predicted_bpb, observed_ceg, predicted_ceg, observed_status, predicted_status
                        (compute-equivalent gain over the 2026 mix at the run's size; status "ok" or why there is none).
ceg_ladder_curve.csv    target, control, ai_share, ceg, status: our law's CEG curve for the 268M, 20 TPP_h ladder.
ceg_ladder_summary.json [CegSummary]: target, human_tpp_range, status counts, the ladder control's predicted/observed CEG.
cost_of_not_filtering.csv  target, n_params, human_tpp, ai_share, label (named share or empty), cost (inf: never reached).
optimal_share.csv       target, n_params, human_tpp, optimal_ratio, optimal_share, largest_loss_reduction (268M).
token_value.csv         target, n_params, human_tpp, ratio, value (average value of the added AI tokens in human tokens).
masking_contrasts.csv   name, control, split, ratio, human_tpp, delta_human_bpb, delta_ai_bpb, critical_share (FW26-H/AI).
masking.json            MaskingSummary: ai_runs, human_harmed, ai_improved, control_ai_loss_reduction_range,
                        median_flip_share, at_share [ai_share, masked, masked_share, median_reported_change], by_split.
"""
