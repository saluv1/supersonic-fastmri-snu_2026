"""Pre-compute foreground masks for every image volume.

The 2026 scorer averages SSIM_full only inside a foreground mask, produced by
`utils.common.metrics.foreground_mask`. Rather than reimplement that thresholding
and morphology (as the 2025 code did, with a brain/knee cutoff split that no
longer applies to this knee-only track), this module calls the scorer's own
function, so the training-time mask can never drift from the graded one.

Masks are cached as .npy next to each other under `--data-path-mask` and reused
on subsequent runs. Not wired into the training loop yet; see the mask-weighted
loss work.
"""

import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from utils.common.metrics import foreground_mask


def mask_filename(mask_dir: Path, volume_name: str, slice_index: int) -> Path:
    stem = volume_name[:-3] if volume_name.endswith('.h5') else volume_name
    return Path(mask_dir) / f'{stem}_s{slice_index}.npy'


def generate_for_file(fname: Path, mask_dir: Path, target_key: str = 'image_label') -> int:
    """Write one .npy per slice; skip slices that already have one."""
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(fname, 'r') as hf:
        num_slices = hf[target_key].shape[0]
        written = 0
        for i in range(num_slices):
            out = mask_filename(mask_dir, Path(fname).name, i)
            if out.exists():
                continue
            np.save(out, foreground_mask(hf[target_key][i]).astype(np.uint8))
            written += 1
    return written


def generate(mask_dir, *data_dirs, target_key: str = 'image_label') -> None:
    """Generate masks for the image volumes under each of `data_dirs`."""
    for data_dir in data_dirs:
        if data_dir is None:
            continue
        files = sorted(Path(data_dir, 'image').glob('*.h5'))
        total = 0
        for fname in tqdm(files, desc=f'foreground masks: {Path(data_dir).name}'):
            total += generate_for_file(fname, mask_dir, target_key)
        print(f'  {Path(data_dir)}: {len(files)} volumes, {total} new slice masks')


def max_mask_area(mask_dir: Path, volume_name: str, num_slices: int) -> float:
    """Largest foreground area across a volume, used to normalise per-slice weights."""
    areas = []
    for i in range(num_slices):
        f = mask_filename(mask_dir, volume_name, i)
        if f.exists():
            areas.append(float(np.load(f).sum()))
    return max(areas) if areas else 0.0