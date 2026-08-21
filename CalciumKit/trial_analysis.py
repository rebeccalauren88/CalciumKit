import numpy as np
import pynapple as nap
from scipy import stats


def ensure_cells_x_time(traces, timestamps):
    return traces.T if traces.shape[0] == len(timestamps) else traces


def extract_trials(traces, timestamps, tone_events, trial_window=50, window_before=0.0, zscore=True):
    """Cut trial windows out of a continuous recording, aligned to each event, via pynapple's compute_perievent.

    Returns:
        trial_data: cells x trials x timepoints
        t_rel: time relative to event onset (starts at -window_before)
    """
    timestamps = np.asarray(timestamps, dtype=float)
    tone_events = np.asarray(tone_events, dtype=float)

    data = ensure_cells_x_time(traces, timestamps)

    tsdframe = nap.TsdFrame(t=timestamps, d=data.T)
    event_ts = nap.Ts(t=tone_events)
    aligned = nap.compute_perievent(tsdframe, event_ts, window=(-window_before, trial_window))

    t_rel = aligned.t
    trial_data = np.transpose(aligned.values, (2, 1, 0))  # (time, events, cells) -> (cells, trials, time)

    if zscore:
        # each cell z-scored against its own collected trial chunks, not the whole trace
        m = np.nanmean(trial_data, axis=(1, 2), keepdims=True)
        s = np.nanstd(trial_data, axis=(1, 2), keepdims=True)
        s[(s == 0) | np.isnan(s)] = 1
        trial_data = (trial_data - m) / s

    return trial_data, t_rel


def compute_event_triggered_etas(
    traces, timestamps, events,
    window_pre=5.0, window_post=10.0,
    baseline_window=(-5.0, -3.0),
    zscore=False,
):
    """Per-cell, baseline-corrected event-triggered averages, aligned via pynapple's compute_perievent.

    traces: (T x N) or (N x T). events: one shared list of event times for all cells.
    zscore: z-score each cell's full trace first, so cells with very different raw amplitude scales
        (e.g. deconvolved spikes) are comparable when averaged/plotted together.
    """
    if traces.ndim != 2:
        raise ValueError("`traces` must be 2D (T x N) or (N x T).")
    if traces.shape[0] == timestamps.shape[0]:
        data = traces.T
    elif traces.shape[1] == timestamps.shape[0]:
        data = traces
    else:
        raise ValueError("`traces` first or second dimension must match len(timestamps).")

    if timestamps.ndim != 1 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("`timestamps` must be a strictly increasing 1D array.")

    if zscore:
        data = stats.zscore(data, axis=1, nan_policy='omit')
        data = np.nan_to_num(data, nan=0.0)

    tsdframe = nap.TsdFrame(t=timestamps, d=data.T)
    event_ts = nap.Ts(t=np.asarray(events, dtype=float))
    aligned = nap.compute_perievent(tsdframe, event_ts, window=(-window_pre, window_post))

    t_rel = aligned.t
    etas_per_cell = np.nanmean(aligned.values, axis=1).T  # (time, events, cells) -> (cells, time), averaged over events

    b0, b1 = baseline_window
    if b0 < -window_pre or b1 > window_post:
        raise ValueError("baseline_window must lie within [-window_pre, window_post].")
    bi = np.where((t_rel >= b0) & (t_rel <= b1))[0]
    if bi.size == 0:
        raise ValueError("baseline_window chosen too narrow; contains no samples.")
    baselines = etas_per_cell[:, bi].mean(axis=1, keepdims=True)
    etas_per_cell = etas_per_cell - baselines

    return t_rel, etas_per_cell


def field_width(tuning_curve, half_max_frac=0.5):
    """Full-width-at-half-max around the peak, in the same units as tuning_curve's index (a pandas Series)."""
    values = tuning_curve.to_numpy()
    baseline = values.min()
    half_max = baseline + (values.max() - baseline) * half_max_frac

    peak_idx = values.argmax()
    above = values >= half_max

    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < len(above) - 1 and above[right + 1]:
        right += 1

    return tuning_curve.index[right] - tuning_curve.index[left]


def trial_reliability(trial_data):
    """Per-cell reliability: each trial's time-course correlated against the trial-average, then averaged.

    trial_data: cells x trials x timepoints (e.g. from extract_trials).
    """
    n_cells, n_trials, n_time = trial_data.shape
    reliability = np.full(n_cells, np.nan)
    for c in range(n_cells):
        trials = trial_data[c]
        mean_trace = np.nanmean(trials, axis=0)
        corrs = [
            np.corrcoef(trials[t], mean_trace)[0, 1]
            for t in range(n_trials) if not np.all(np.isnan(trials[t]))
        ]
        if corrs:
            reliability[c] = np.nanmean(corrs)
    return reliability


def t_confidence_interval(across_cells, alpha=0.05):
    """across_cells: (Nc x T). Returns low, high arrays shape (T,)."""
    m = np.nanmean(across_cells, axis=0)
    s = stats.sem(across_cells, axis=0, nan_policy='omit')
    nc = across_cells.shape[0]
    if nc < 2:
        return m, m
    tcrit = stats.t.ppf(1 - alpha / 2.0, df=nc - 1)
    return m - tcrit * s, m + tcrit * s


def bootstrap_confidence_interval(across_cells, alpha=0.05, n_boot=1000, rng=None):
    """across_cells: (Nc x T). Nonparametric bootstrap over cells."""
    rng = np.random.default_rng(None if rng is None else rng)
    Nc, T = across_cells.shape
    if Nc < 2:
        m = np.nanmean(across_cells, axis=0)
        return m, m
    boots = np.empty((n_boot, T), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, Nc, Nc)
        boots[b] = np.nanmean(across_cells[idx], axis=0)
    low = np.percentile(boots, 100 * alpha / 2.0, axis=0)
    high = np.percentile(boots, 100 * (1 - alpha / 2.0), axis=0)
    return low, high


def contiguous_sig_runs(ci_low, ci_high, min_duration_s, t_rel):
    """Segments where the CI excludes 0 for at least min_duration_s. Returns (start_idx, end_idx) pairs, inclusive."""
    dt = np.median(np.diff(t_rel))
    min_len = max(1, int(np.round(min_duration_s / dt)))
    sig_mask = (ci_low > 0) | (ci_high < 0)

    runs = []
    start = None
    for i, v in enumerate(sig_mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    if start is not None and (len(sig_mask) - start) >= min_len:
        runs.append((start, len(sig_mask) - 1))
    return runs


def eta_all(
    dFF, timestamps, events,
    window_pre=5.0, window_post=10.0,
    ci="tci", alpha=0.05, n_boot=1000,
    baseline_window=(-5.0, -3.0),
    sig_duration=10.0,
    zscore=False,
):
    """Per-cell baseline-corrected ETAs, group mean, CI, and significant segments."""
    t_rel, across_eta_per_cell = compute_event_triggered_etas(
        dFF, timestamps, events,
        window_pre=window_pre, window_post=window_post,
        baseline_window=baseline_window,
        zscore=zscore,
    )

    mean_trace = np.nanmean(across_eta_per_cell, axis=0)

    if ci == "tci":
        lo, hi = t_confidence_interval(across_eta_per_cell, alpha=alpha)
    elif ci == "bci":
        lo, hi = bootstrap_confidence_interval(across_eta_per_cell, alpha=alpha, n_boot=n_boot)
    else:
        raise ValueError("ci must be 'tci' or 'bci'")

    sig_runs = contiguous_sig_runs(lo, hi, min_duration_s=sig_duration, t_rel=t_rel)
    return t_rel, mean_trace, (lo, hi), sig_runs, across_eta_per_cell
