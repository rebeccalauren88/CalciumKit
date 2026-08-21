import json
import re
import h5py
import numpy as np
import tifffile

# sampling constants, fixed for this cohort's Femtonics rig
CURVE_SAMPLE_DT_MS = 0.06484
FRAME_DT_MS = 42.535039999999995
TRUE_FRAME_DT_MS = FRAME_DT_MS * 2
TRUE_FRAME_RATE = 1000.0 / TRUE_FRAME_DT_MS
CH0_T0_MS = 0.0
CH1_T0_MS = FRAME_DT_MS


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
        date = f'{date_raw[4:8]}-{date_raw[0:2]}-{date_raw[2:4]}'
    else:
        subject, session, date = stem, 'unknown', 'unknown'
    return subject, session, date


def extract_mesc_file(mesc_path, out_root):
    """Extract both channels of a raw .mesc into metadata.json, frame timestamps, and per-channel tiffs."""
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
        print(f'  ✓ frame_times_ms_green.npy  ({true_n_frames} frames, {ch0_frame_times_ms[-1] / 1000:.1f} s)')
        print(f'  ✓ frame_times_ms_red.npy    (offset +{FRAME_DT_MS:.3f} ms from green)')

        # 3. imaging data
        channel_parity = {
            'Channel_0': (0, 'even', 'green'),
            'Channel_1': (1, 'odd', 'red'),
        }

        for ch_name, (parity, parity_label, color) in channel_parity.items():
            if ch_name not in unit:
                print(f'  skipping {ch_name} — not found')
                continue

            print(f'\n  Loading {ch_name} ({color})...')
            raw = unit[ch_name][:]
            data = raw[parity::2]

            active_mean = raw[parity::2].mean()
            blanked_mean = raw[1 - parity::2].mean()
            ratio = active_mean / blanked_mean if blanked_mean > 0 else float('inf')

            if ratio > 1.05:
                print(f'  ✓ parity check passed  (active {active_mean:.1f}  blanked {blanked_mean:.1f}  ratio {ratio:.2f}x)')
            else:
                print(f'  ⚠ parity check FAILED  (active {active_mean:.1f}  blanked {blanked_mean:.1f}  ratio {ratio:.2f}x) — verify visually')

            t0_ms = CH0_T0_MS if parity == 0 else CH1_T0_MS
            out_path = out_dir / f'channel_{color}.tif'
            tifffile.imwrite(
                out_path,
                data,
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
                    })
                }
            )
            print(f'  ✓ channel_{color}.tif  shape={data.shape}  dtype={data.dtype}  {out_path.stat().st_size / 1e9:.2f} GB')

    print(f'\n  Output files:')
    for p in sorted(out_dir.iterdir()):
        print(f'    {p.name:<45}  {p.stat().st_size / 1e6:8.1f} MB')

    return out_dir
