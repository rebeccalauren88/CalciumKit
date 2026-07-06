"""
Femtonics .mesc -> tiff -> Suite2p batch pipeline

Walks a cohort root folder for Day1..Day4 (or any Day* folders), and for
every .mesc file inside each:
  1. Extracts Channel_0 (green) and Channel_1 (red) to BigTIFF, streamed
     in blocks so it never loads a whole ~18GB channel into RAM at once.
  2. Runs Suite2p (registration + detection + deconvolution) on both
     channels separately, using the frame rate computed during extraction.

Requires: h5py, numpy, tifffile, suite2p
Tested against suite2p's current db/settings API (default_db/default_settings).
torch_device is set to 'mps' for Apple Silicon; falls back to CPU automatically
if MPS isn't available.
"""

from pathlib import Path
import json
import re

import h5py
import numpy as np
import tifffile

from suite2p import run_s2p, default_settings, default_db


# ── configure this ───────────────────────────────────────────────────────────
COHORT_ROOT = Path(
    '/Volumes/rkc_ramirezlab/Home/suthardr/Projects/2photon_menace/'
    'FriendsCohort/friends_cohort_june2026'
)
DAY_FOLDERS = sorted(COHORT_ROOT.glob('Day*'))   # picks up Day1.../Day2-TFC/Day3.../Day4-RecallToneB

RUN_SUITE2P = True          # set False to only extract tiffs this run
TORCH_DEVICE = 'cpu'        # Apple Silicon; falls back to CPU if unavailable
BLOCK_FRAMES = 2000         # streaming block size for extraction (tune for RAM/speed)

# Set this to a single .mesc path to test the whole pipeline on one file only.
# Leave as None to run the full Day1-4 batch.
# TEST_FILE = None
TEST_FILE = Path(
    '/Users/suthardr/Desktop/code_test/friends_chandler_day2_06162026.mesc'
)

# ── sampling constants (fixed for this cohort) ──────────────────────────────
CURVE_SAMPLE_DT_MS = 0.06484
FRAME_DT_MS        = 42.535039999999995
TRUE_FRAME_DT_MS   = FRAME_DT_MS * 2
TRUE_FRAME_RATE    = 1000.0 / TRUE_FRAME_DT_MS
CH0_T0_MS          = 0.0
CH1_T0_MS          = FRAME_DT_MS


# ── extraction helpers ───────────────────────────────────────────────────────
def find_imaging_unit(f):
    best_path, best_frames, best_shape = None, 0, None

    def visit(name, obj):
        nonlocal best_path, best_frames, best_shape
        if isinstance(obj, h5py.Dataset) and 'Channel_0' in name:
            shape = obj.shape
            if len(shape) == 3 and shape[0] > best_frames:
                best_frames = shape[0]
                best_path = name.replace('/Channel_0', '')
                best_shape = shape

    f.visititems(visit)
    return best_path, best_frames, best_shape


def parse_filename(stem):
    match = re.match(r'friends_(\w+)_(day\d+)_(\d{8})', stem)
    if match:
        subject = match.group(1)
        session = match.group(2)
        date_raw = match.group(3)
        date = f'20{date_raw[4:6]}-{date_raw[0:2]}-{date_raw[2:4]}'
    else:
        subject, session, date = stem, 'unknown', 'unknown'
    return subject, session, date


def channel_frame_generator(dset, parity, stats, block_frames=BLOCK_FRAMES):
    """Yields active-parity frames one at a time, reading only `block_frames`
    raw frames from disk/NAS at once. Also accumulates the parity-check
    stats (active vs blanked mean) into the `stats` dict, populated once
    the generator is fully consumed."""
    active_sum, active_n = 0.0, 0
    blanked_sum, blanked_n = 0.0, 0
    for start in range(0, dset.shape[0], block_frames):
        block = dset[start:start + block_frames]
        active = block[parity::2]
        blanked = block[1 - parity::2]
        active_sum += float(active.sum())
        active_n += active.size
        blanked_sum += float(blanked.sum())
        blanked_n += blanked.size
        for frame in active:
            yield frame
    stats['active_mean'] = active_sum / active_n if active_n else float('nan')
    stats['blanked_mean'] = blanked_sum / blanked_n if blanked_n else float('nan')


def extract_mesc_file(mesc_path, out_root):
    """Extracts both channels from one .mesc file to BigTIFF + metadata.json
    + frame timestamps. Returns the OUT_DIR path, or None if skipped."""
    stem = mesc_path.stem
    subject, session, date = parse_filename(stem)
    out_dir = out_root / f'{stem}_extracted'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"═" * 70}')
    print(f'  {subject.upper()}  |  {session}  |  {date}')
    print(f'  {mesc_path.name}')
    print(f'  → {out_dir}')
    print(f'{"═" * 70}')

    with h5py.File(mesc_path, 'r') as f:
        unit_path, n_frames, frame_shape = find_imaging_unit(f)
        if unit_path is None:
            print('  ⚠ no imaging MUnit found — skipping')
            return None

        true_n_frames = n_frames // 2
        H, W = frame_shape[1], frame_shape[2]

        print(f'  Imaging unit:  {unit_path}')
        print(f'  Stored frames: {n_frames}  →  {true_n_frames} true frames')
        print(f'  Frame size:    {H} × {W} px  |  {TRUE_FRAME_RATE:.2f} Hz\n')

        unit = f[unit_path]

        # 1. metadata
        meta = {
            'file': str(mesc_path),
            'subject': subject,
            'session': session,
            'date': date,
            'munit': {
                'path': unit_path,
                'section_type': str(unit.attrs.get('SectionTypeDebugString', 'unknown')),
                'technology': str(unit.attrs.get('TechnologyTypeDebugString', 'unknown')),
                'scanner': str(unit.attrs.get('Scanner', 'unknown')),
                'n_channels': int(unit.attrs.get('VecChannelsSize', 2)),
                'stored_frame_period_ms': FRAME_DT_MS,
                'stored_frame_rate_hz': 1000.0 / FRAME_DT_MS,
                'stored_n_frames': n_frames,
                'true_frame_period_ms': TRUE_FRAME_DT_MS,
                'true_frame_rate_hz': TRUE_FRAME_RATE,
                'true_n_frames': true_n_frames,
                'frame_size_px': [H, W],
                'pmt_gating': 'interleaved_channels',
                'channel_0_active_parity': 'even (0-indexed)',
                'channel_1_active_parity': 'odd (0-indexed)',
                'channel_0_t0_ms': CH0_T0_MS,
                'channel_1_t0_ms': CH1_T0_MS,
                'channel_temporal_offset_ms': FRAME_DT_MS,
                'T0_ms': float(unit.attrs.get('T0InMs', 0.0)),
                'precalc_time_ms': float(unit.attrs.get('PrecalculationTimeInMs', 0.0)),
                'mes_version': json.loads(unit.attrs['VersionInfoJSON'])['mesVersion']
                               if 'VersionInfoJSON' in unit.attrs else 'unknown',
                'uuid': unit.attrs['Uuid'].tolist() if 'Uuid' in unit.attrs else None,
            },
            'channels': {},
        }
        for ch_name in ['Channel_0', 'Channel_1']:
            if ch_name in unit:
                meta['channels'][ch_name] = {
                    k: (v.tolist() if hasattr(v, 'tolist') else str(v))
                    for k, v in dict(unit[ch_name].attrs).items()
                }
        with open(out_dir / 'metadata.json', 'w') as fj:
            json.dump(meta, fj, indent=2, default=str)
        print(f'  ✓ metadata.json')

        # 2. frame timestamps
        t0 = float(unit.attrs.get('T0InMs', 0.0))
        ch0_frame_times_ms = t0 + CH0_T0_MS + np.arange(true_n_frames) * TRUE_FRAME_DT_MS
        ch1_frame_times_ms = t0 + CH1_T0_MS + np.arange(true_n_frames) * TRUE_FRAME_DT_MS
        np.save(out_dir / 'frame_times_ms_green.npy', ch0_frame_times_ms)
        np.save(out_dir / 'frame_times_ms_red.npy', ch1_frame_times_ms)
        print(f'  ✓ frame_times_ms_green.npy  ({true_n_frames} frames, {ch0_frame_times_ms[-1]/1000:.1f} s)')
        print(f'  ✓ frame_times_ms_red.npy    (offset +{FRAME_DT_MS:.3f} ms from green)')

        # 3. imaging data (streamed, chunked)
        channel_parity = {
            'Channel_0': (0, 'even', 'green'),
            'Channel_1': (1, 'odd', 'red'),
        }
        for ch_name, (parity, parity_label, color) in channel_parity.items():
            if ch_name not in unit:
                print(f'  skipping {ch_name} — not found')
                continue

            print(f'\n  Streaming {ch_name} ({color})...')
            dset = unit[ch_name]
            stats = {}
            out_path = out_dir / f'channel_{color}.tif'

            t0_ms = CH0_T0_MS if parity == 0 else CH1_T0_MS
            tifffile.imwrite(
                out_path,
                channel_frame_generator(dset, parity, stats),
                shape=(true_n_frames, H, W),
                dtype=dset.dtype,
                bigtiff=True,
                metadata={
                    'axes': 'TYX',
                    'fps': TRUE_FRAME_RATE,
                    'unit': 'um',
                    'Info': json.dumps({
                        'subject': subject,
                        'session': session,
                        'date': date,
                        'channel': ch_name,
                        'color': color,
                        'frame_rate_hz': TRUE_FRAME_RATE,
                        'frame_period_ms': TRUE_FRAME_DT_MS,
                        'n_frames': true_n_frames,
                        'n_frames_raw': n_frames,
                        'pmt_gating': 'interleaved_channels',
                        'active_frame_parity': parity_label + ' (0-indexed)',
                        't0_ms': t0_ms,
                        'source_file': str(mesc_path),
                        'source_path': unit_path + '/' + ch_name,
                    }),
                },
            )

            ratio = stats['active_mean'] / stats['blanked_mean'] if stats['blanked_mean'] else float('inf')
            if ratio > 1.05:
                print(f'  ✓ parity check passed  '
                      f'(active {stats["active_mean"]:.1f}  blanked {stats["blanked_mean"]:.1f}  ratio {ratio:.2f}x)')
            else:
                print(f'  ⚠ parity check FAILED  '
                      f'(active {stats["active_mean"]:.1f}  blanked {stats["blanked_mean"]:.1f}  '
                      f'ratio {ratio:.2f}x) — verify visually')

            print(f'  ✓ {out_path.name}  {out_path.stat().st_size/1e9:.2f} GB')

    return out_dir


# ── suite2p helpers ──────────────────────────────────────────────────────────
def verify_tiff_matches_metadata(tif_path, expected_frames):
    with tifffile.TiffFile(tif_path) as tif:
        shape = tif.series[0].shape
    n_frames_on_disk = shape[0] if len(shape) == 3 else 1
    if n_frames_on_disk != expected_frames:
        print(f'  ⚠ FRAME COUNT MISMATCH for {tif_path.name}: '
              f'metadata says {expected_frames}, tiff on disk has {n_frames_on_disk}')
        return False
    return True


def run_suite2p_on_channel(tif_path, fs, out_root):
    save_dir = out_root / f'{tif_path.stem}_suite2p'
    save_dir.mkdir(parents=True, exist_ok=True)

    db = default_db()
    db['data_path'] = [str(tif_path.parent)]
    db['file_list'] = [tif_path.name]
    db['save_path0'] = str(save_dir)
    db['input_format'] = 'tif'
    db['nplanes'] = 1
    db['nchannels'] = 1

    settings = default_settings()
    settings['fs'] = fs
    settings['torch_device'] = TORCH_DEVICE
    settings['run']['do_registration'] = 1
    settings['run']['do_detection'] = True
    settings['run']['do_deconvolution'] = True
    settings['io']['save_mat'] = False

    print(f'  Running suite2p on {tif_path.name}  (fs={fs:.3f} Hz, device={TORCH_DEVICE})')
    run_s2p(db=db, settings=settings)
    print(f'  ✓ Finished: {save_dir}')


def run_suite2p_on_folder(out_dir):
    meta_path = out_dir / 'metadata.json'
    with open(meta_path) as f:
        meta = json.load(f)
    fs = meta['munit']['true_frame_rate_hz']
    expected_frames = meta['munit']['true_n_frames']

    suite2p_out_root = out_dir / 'suite2p_outputs'
    suite2p_out_root.mkdir(exist_ok=True)

    for color in ['green', 'red']:
        tif_path = out_dir / f'channel_{color}.tif'
        if not tif_path.exists():
            print(f'  skipping channel_{color}.tif — not found')
            continue
        if not verify_tiff_matches_metadata(tif_path, expected_frames):
            print(f'  ⚠ skipping suite2p run for {tif_path.name} until this is resolved')
            continue
        run_suite2p_on_channel(tif_path, fs, suite2p_out_root)


# ── main batch loop ──────────────────────────────────────────────────────────
def main():
    # ── single-file test mode ────────────────────────────────────────────
    if TEST_FILE is not None:
        if not TEST_FILE.exists():
            print(f'TEST_FILE not found: {TEST_FILE}')
            return
        print(f'TEST MODE — running the full pipeline on one file only:\n  {TEST_FILE}')
        try:
            out_dir = extract_mesc_file(TEST_FILE, out_root=TEST_FILE.parent)
            if out_dir is not None and RUN_SUITE2P:
                run_suite2p_on_folder(out_dir)
            print(f'\n{"═" * 70}')
            print('TEST MODE complete. Check the output above before running the full batch.')
        except Exception as e:
            print(f'  ✗ FAILED: {TEST_FILE.name}: {e}')
        return

    # ── full Day1-4 batch ────────────────────────────────────────────────
    if not DAY_FOLDERS:
        print(f'No Day* folders found under {COHORT_ROOT}')
        return

    print(f'Found {len(DAY_FOLDERS)} day folder(s): {[d.name for d in DAY_FOLDERS]}')

    failed = []
    processed = 0

    for day_folder in DAY_FOLDERS:
        mesc_files = sorted(day_folder.glob('*.mesc'))
        if not mesc_files:
            print(f'\n[{day_folder.name}] no .mesc files found — skipping')
            continue

        print(f'\n{"#" * 70}')
        print(f'  {day_folder.name}  —  {len(mesc_files)} .mesc file(s)')
        print(f'{"#" * 70}')

        for mesc_path in mesc_files:
            try:
                out_dir = extract_mesc_file(mesc_path, out_root=day_folder)
                if out_dir is None:
                    continue
                if RUN_SUITE2P:
                    run_suite2p_on_folder(out_dir)
                processed += 1
            except Exception as e:
                print(f'  ✗ FAILED: {mesc_path.name}: {e}')
                failed.append((f'{day_folder.name}/{mesc_path.name}', str(e)))

    print(f'\n{"═" * 70}')
    print(f'Done. {processed} file(s) processed successfully, {len(failed)} failed.')
    if failed:
        print('Failed:')
        for name, err in failed:
            print(f'  {name}: {err}')


if __name__ == '__main__':
    main()
