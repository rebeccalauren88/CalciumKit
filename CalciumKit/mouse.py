from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
from scipy.signal import savgol_filter

from CalciumKit.femtonics import parse_filename


NEUROPIL_COEF = 0.7
BASELINE_WINDOW = 501
BASELINE_POLYORDER = 3


class Suite2pData:
    def __init__(self, suite2p_dir):
        self.dir = suite2p_dir
        self.F = np.load(suite2p_dir / 'F.npy', allow_pickle=True)
        self.Fneu = np.load(suite2p_dir / 'Fneu.npy', allow_pickle=True)
        self.spks = np.load(suite2p_dir / 'spks.npy', allow_pickle=True)
        self.stat = np.load(suite2p_dir / 'stat.npy', allow_pickle=True)
        self.iscell = np.load(suite2p_dir / 'iscell.npy', allow_pickle=True)

    @cached_property
    def is_cell(self):
        return self.iscell[:, 0].astype(bool)

    @cached_property
    def dff(self):
        """Neuropil-corrected dF/F0, one row per ROI (same rows as F/stat/iscell)."""
        f_corrected = self.F - NEUROPIL_COEF * self.Fneu

        window_length = min(BASELINE_WINDOW, f_corrected.shape[1] - 1)
        if window_length % 2 == 0:
            window_length -= 1
        baseline = savgol_filter(f_corrected, window_length=window_length, polyorder=BASELINE_POLYORDER, axis=1)
        baseline[baseline == 0] = np.nan

        dff = (f_corrected - baseline) / baseline
        return np.nan_to_num(dff, nan=0.0, posinf=0.0, neginf=0.0)


def _find_suite2p_dir(extracted_dir, color):
    for stat_path in extracted_dir.rglob('stat.npy'):
        if any(color in part.lower() for part in stat_path.parts):
            return stat_path.parent
    return None


class TrialStructure:
    """Tone/trace/shock timing for a fear conditioning session, and the epoch windows used for imaging analysis."""

    def __init__(self, tone_times, tone_dur=20, trace_dur=20, shock_offset=39, us_window=10, iti_offset=41, iti_dur=120):
        self.tone_times = np.array(tone_times, dtype=float)
        self.tone_dur = tone_dur
        self.trace_dur = trace_dur
        self.shock_offset = shock_offset  # seconds after tone onset; matches the behavior notebook's Bpod shock times (tone + 39s)
        self.us_window = us_window  # imaging response-capture window after shock onset, not the shock's own duration
        self.iti_offset = iti_offset  # seconds after tone onset; matches the behavior notebook's ITI start (shock_offset + 2)
        self.iti_dur = iti_dur

    @property
    def trace_times(self):
        return self.tone_times + self.tone_dur

    @property
    def shock_times(self):
        return self.tone_times + self.shock_offset

    @property
    def epochs(self):
        return {
            'CS+': (0, self.tone_dur),
            'Trace': (self.tone_dur, self.tone_dur + self.trace_dur),
            'US': (self.shock_offset, self.shock_offset + self.us_window),
            'ITI': (self.iti_offset, self.iti_offset + self.iti_dur),
        }


TFC_STRUCTURE = TrialStructure(tone_times=[120, 280, 440, 600, 760])

TRIAL_STRUCTURES = {
    'tfc': TFC_STRUCTURE,
    'recallA': TFC_STRUCTURE,
    'recallB': TFC_STRUCTURE,
}


class Session:
    def __init__(self, mouse_id, day, date, mesc_path, processed_root, behavior_root):
        self.mouse_id = mouse_id
        self.day = day
        self.date = date
        self.mesc_path = mesc_path
        self.processed_root = processed_root
        self.behavior_root = behavior_root
        self.registration = None
        self.task = None

    @property
    def trial_structure(self):
        return TRIAL_STRUCTURES.get(self.task)

    @cached_property
    def extracted_dir(self):
        return self.processed_root / f'{self.mesc_path.stem}_extracted'

    @cached_property
    def frame_times_green(self):
        return np.load(self.extracted_dir / 'frame_times_ms_green.npy')

    @cached_property
    def frame_times_red(self):
        return np.load(self.extracted_dir / 'frame_times_ms_red.npy')

    @cached_property
    def green(self):
        suite2p_dir = _find_suite2p_dir(self.extracted_dir, 'green')
        return Suite2pData(suite2p_dir) if suite2p_dir else None

    @cached_property
    def red(self):
        suite2p_dir = _find_suite2p_dir(self.extracted_dir, 'red')
        return Suite2pData(suite2p_dir) if suite2p_dir else None

    def tsd(self, color, signal='dff'):
        """The given channel's signal ('dff' or 'spks'), is_cell-filtered, as a pynapple TsdFrame in seconds."""
        s2p = self.green if color == 'green' else self.red
        frame_times_ms = self.frame_times_green if color == 'green' else self.frame_times_red
        data = getattr(s2p, signal)[s2p.is_cell]
        frame_times_ms = frame_times_ms[:data.shape[1]]
        time_s = (frame_times_ms - frame_times_ms[0]) / 1000.0
        return nap.TsdFrame(t=time_s, d=data.T)

    @cached_property
    def behavior(self):
        if self.behavior_root is None:
            return None
        date_compact = self.date.replace('-', '')
        matches = [
            p for p in self.behavior_root.rglob('processedRotary_*.csv')
            if date_compact in p.name and self.mouse_id.lower() in p.name.lower()
        ]
        return pd.read_csv(matches[0]) if matches else None

    @cached_property
    def behavior_tsd(self):
        df = self.behavior
        if df is None:
            return None
        cols = [c for c in df.columns if c != 'time_s']
        return nap.TsdFrame(t=df['time_s'].to_numpy(), d=df[cols].to_numpy(), columns=cols)


class Mouse:
    def __init__(self, mouse_id, raw_root, processed_root, behavior_root=None):
        self.mouse_id = mouse_id
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.behavior_root = Path(behavior_root) if behavior_root else None
        self.sessions = self._discover_sessions()

    def _discover_sessions(self):
        sessions = {}
        for mesc_path in sorted((self.raw_root / self.mouse_id).glob('*.mesc')):
            subject, day, date = parse_filename(mesc_path.stem)
            if day == 'unknown':
                continue
            sessions[day] = Session(
                self.mouse_id, day, date, mesc_path,
                self.processed_root, self.behavior_root,
            )
        return sessions
