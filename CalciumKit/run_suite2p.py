"""Organize per-channel tiffs from mesc extraction into suite2p_<color> folders, run suite2p on each, then open the GUI to check segmentation."""

from pathlib import Path
import shutil
import subprocess
import sys

import suite2p

BASE_OPS = suite2p.default_ops()

# red/jRGECO1a neurons: sparse, discrete somata - correlation-based detection works well.
# tau matters here since neurons use the deconvolved `spks` trace downstream.
NEURON_OPS = {**BASE_OPS,
    'fs': 11.7,               # imaging frame rate (Hz) - set to match this session
    'tau': 1.5,               # jRGECO1a decay time constant (s)
    'nchannels': 1,           # each color runs separately
    'diameter': 12,           # expected soma diameter (px)
    'sparse_mode': True,
    'threshold_scaling': 1.0,
}

# green/GCaMP6f astrocytes: tiled, overlapping processes - anatomical (Cellpose-style) detection
# on the mean image usually segments somata better than activity-correlation detection.
# tau barely matters since astrocytes only ever use `dff`, not `spks`, downstream.
ASTROCYTE_OPS = {**BASE_OPS,
    'fs': 11.7,               # imaging frame rate (Hz) - set to match this session
    'tau': 0.7,               # GCaMP6f decay time constant (s), unused by dff-only analysis
    'nchannels': 1,           # each color runs separately
    'sparse_mode': False,
    'anatomical_only': 3,     # Cellpose-style detection on the mean image
    'threshold_scaling': 0.75,  # more permissive - astrocyte signal is lower-amplitude/diffuse
    'pretrained_model': 'cpsam',  # Cellpose-SAM - check this key exists in your suite2p version
    'diameter': 0,            # let cpsam auto-estimate size instead of forcing a fixed diameter
    'gpu': True,              # cpsam is much slower than cyto models on CPU
    'flow_threshold': 0.4,
    'cellprob_threshold': 0.0,
}

COLOR_OPS = {'red': NEURON_OPS, 'green': ASTROCYTE_OPS}


def organize_channel(extracted_dir, color):
    """Move channel_<color>.tif into its own suite2p_<color> folder, if not already there."""
    dest_dir = extracted_dir / f'suite2p_{color}'
    dest_dir.mkdir(exist_ok=True)
    src = extracted_dir / f'channel_{color}.tif'
    dest = dest_dir / src.name
    if src.exists() and not dest.exists():
        shutil.move(str(src), str(dest))
    return dest_dir


def run_channel(dest_dir, ops):
    db = {'data_path': [str(dest_dir)], 'save_path0': str(dest_dir)}
    suite2p.run_s2p(ops=ops, db=db)


def open_gui():
    subprocess.run([sys.executable, '-m', 'suite2p'])


def process(extracted_dir, ops_overrides=None, colors=('green', 'red'), launch_gui=True):
    extracted_dir = Path(extracted_dir)

    for color in colors:
        ops = {**COLOR_OPS[color], **(ops_overrides or {})}
        dest_dir = organize_channel(extracted_dir, color)
        print(f'Running suite2p on {color} -> {dest_dir}')
        run_channel(dest_dir, ops)

    if launch_gui:
        print('Opening suite2p GUI - load stat.npy from suite2p_<color>/suite2p/plane0/ to check segmentation.')
        open_gui()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('extracted_dir', help='folder with channel_green.tif / channel_red.tif from femtonics.extract_mesc_file')
    parser.add_argument('--no-gui', action='store_true', help='skip opening the suite2p GUI after processing')
    args = parser.parse_args()
    process(args.extracted_dir, launch_gui=not args.no_gui)
