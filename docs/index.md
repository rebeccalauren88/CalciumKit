# CalciumKit pipeline

## Imaging

`mesc_to_tiff.ipynb` exports a raw Femtonics `.mesc` into per-channel tiffs. Set `MESC_FOLDER` (raw `.mesc` files) and `OUT_ROOT` (where extracted output goes), then it calls `extract_mesc_file` (from `CalciumKit.femtonics`) on every `.mesc` file in `MESC_FOLDER`.

For each file, `extract_mesc_file` writes a `<name>_extracted/` folder containing:
- `metadata.json` — subject/session/date, frame rate, frame size, imaging unit info.
- `frame_times_ms_green.npy` / `frame_times_ms_red.npy` — per-channel frame timestamps (the two channels are offset by `FRAME_DT_MS` from each other).
- `channel_green.tif` / `channel_red.tif` — the two channels, ready to load into suite2p separately.

Suite2p is run manually per channel from here, with manual curation of the segmentation — not automated by this pipeline.

## Behavior

Raw rotary CSVs (`time_s`, `position_deg`) go through two notebooks in `CalciumKit/rotary/`:

**`rotary_analysis.ipynb`** — raw → processed. Set `input_dir` to the session's raw CSV folder, then it calls `process_rotary_csv` (from `CalciumKit.behavior`) on every raw CSV in that folder. Unwraps position, fits a smoothing spline, derives velocity/acceleration, and writes `processedRotary_<name>.csv` into `input_dir/processed/`.

**`rotary_plotting.ipynb`** — processed → plots. Reads the `processedRotary_*.csv` files and writes PNGs/SVGs into `processed/plots/`:
- `plot_full_session_plotnine()` — all-mice average velocity, 95% CI, tone/trace/shock/ITI shading.
- `plot_full_session()` — same, per individual mouse.
- Shock-triggered ETA plot.
- Epoch-summary table (`mean_in_epoch`, also from `CalciumKit.behavior`) — per-mouse mean velocity in each epoch.
- Tone/trace/shock faceted ETA plot.

`rotary_behavioral_analysis_rebecca.ipynb` is separate — it runs stats (RM ANOVA + Holm posthoc, paired t-tests) on long-form summary data across sessions, not part of the per-session raw-to-plots flow above.

`behavior.py` also has `wheel_kinematics` (Kalman-smoothed wheel motion) — a different, unrelated approach, not used by the rotary notebooks above.

## Mouse / Session

`CalciumKit.mouse` ties imaging and behavior together per animal, without needing to know folder-naming conventions by hand.

`Mouse(mouse_id, raw_root, processed_root, behavior_root=None)` scans `raw_root/<mouse_id>/*.mesc`, parses each filename for day/date, and builds `.sessions` — a `{day: Session}` dict. Files that don't match the expected naming (test scans, oddly-named recordings) are silently skipped.

`Session` — everything lazy, nothing loads until accessed:
- `.mesc_path`, `.day`, `.date`
- `.extracted_dir` — the `<mesc_stem>_extracted/` folder under `processed_root`, from `extract_mesc_file`.
- `.frame_times_green` / `.frame_times_red`
- `.green` / `.red` — a `Suite2pData` object (finds whichever subfolder has "green"/"red" in its path and contains `stat.npy`, so it doesn't matter how suite2p nested its own `plane0/` folder or exactly how the folder was named). `None` if that channel hasn't been curated yet.
- `.behavior` — the matching `processedRotary_*.csv` from `behavior_root`, matched by mouse + date, as a pandas DataFrame. `None` if no behavior file exists for that day.
- `.behavior_tsd` — the same data as a pynapple `TsdFrame` (`time_s` as the time index, the rest of the columns as data). `None` when `.behavior` is `None`.
- `.tsd(color, signal='dff')` — a channel's `'dff'` or `'spks'`, already filtered to `is_cell`, as a pynapple `TsdFrame` indexed in seconds (t0-relative). Replaces manually filtering by `is_cell` and converting frame times from ms.
- `.registration` — `None` until ROICaT produces a cross-day cell-matching table for this mouse.
- `.task` — plain settable string (e.g. `"tfc"`), not inferred automatically.
- `.trial_structure` — looks up `.task` in `TRIAL_STRUCTURES`; `None` until `.task` is set.

`Suite2pData` (on `.green`/`.red`) holds the raw suite2p arrays (`F`, `Fneu`, `spks`, `stat`, `iscell`) plus two derived properties:
- `.is_cell` — boolean mask from `iscell`.
- `.dff` — neuropil-corrected (`F - 0.7*Fneu`) dF/F0, using a Savitzky-Golay rolling baseline (window 501, polyorder 3) — matches the value the original 2P analysis notebook computed for astrocytes.

`TrialStructure` (in `CalciumKit.mouse`) holds tone/trace/shock/ITI timing and the CS+/Trace/US/ITI epoch windows for a fear-conditioning session. `TFC_STRUCTURE` is the known TFC timing (tone onsets `[120, 280, 440, 600, 760]`, spaced 160s apart = tone(20s) + trace(20s) + ITI(120s), with a 120s baseline before the first tone; shock at tone+39s and ITI at tone+41s — both match the behavior notebook's Bpod-recorded times, not derived from `tone_dur`/`trace_dur`/`us_window`). `TRIAL_STRUCTURES` maps `"tfc"`/`"recallA"`/`"recallB"` all to `TFC_STRUCTURE`, since recall sessions reuse the same timing.

## Trial analysis

`CalciumKit.trial_analysis` — generic functions for event-locked analysis, independent of `Mouse`/`Session` (they just take `traces`/`timestamps`/event-time arrays, so they work with any signal, not only suite2p output). The alignment step (cutting/averaging windows around events) is implemented with pynapple's `compute_perievent` under the hood, verified against real Chandler day2 data to reproduce the same numbers as the original hand-rolled interpolation it replaced:
- `extract_trials(traces, timestamps, tone_events, ...)` — cuts trial windows out of a continuous recording, aligned to each event. Returns `(cells x trials x timepoints, t_rel)`.
- `eta_all(traces, timestamps, events, ...)` — per-cell baseline-corrected event-triggered averages, group mean, confidence interval (`t_confidence_interval` or `bootstrap_confidence_interval`), and significant time segments (`contiguous_sig_runs`). This is the function behind the shock-response ETA plots.

Plotting, figure styling, and the CS+/Trace/US-specific pipeline orchestration that used to live in `2P_ANALYSIS.ipynb` were dropped — those functions are kept lean and reusable here rather than bundled into one big pipeline notebook.

`Session.tsd()`/`.behavior_tsd` (see Mouse / Session above) are the pieces that would feed the eventual velocity-alignment work — `eta_all`/`extract_trials` take plain arrays, but a `TsdFrame`'s `.values`/`.t` are already the arrays they expect, and `nap.compute_perievent` works directly against `Session.tsd(...)` if you want lower-level access than `eta_all`.
