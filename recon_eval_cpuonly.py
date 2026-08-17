"""CPU-only reconstruction + evaluation (For running server).

Scoring is identical to recon_eval.py — the same SSIM_full / SSIM_bbox
functions from utils/common/metrics.py are used unchanged.  The only
differences are:

  1. All computation runs on CPU (--GPU_NUM is accepted but ignored).
  2. Per-slice timing is NOT measured; Recon Time is reported as N/A.
  3. CPU threads are capped (default 2) to avoid starving a concurrent
     GPU training run's data-loader.

This script is provided solely for teams whose GPU is occupied by a
long-running training job.  The SSIM scores it produces are valid for
leaderboard upload; the Recon Time field should be left blank (the
organiser will re-time with recon_eval.py on submission).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Hide GPU from this process BEFORE importing torch ─────────────────
# Training runs in a separate process and is unaffected.  This ensures
# torch.cuda.is_available() == False everywhere, which prevents
# center_crop (utils/common/utils.py) from silently moving tensors to
# the GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── Make this process the first to be killed on RAM pressure ──────────
# If system RAM runs low, the Linux OOM killer will terminate this
# eval process instead of the concurrent GPU training process.
try:
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")
except (PermissionError, FileNotFoundError):
    pass  # non-Linux or no permission — skip silently

import torch

# ── Cap CPU threads BEFORE any heavy computation ──────────────────────
# Keep this early so NumPy/MKL/OpenMP also respect the limit.

def _apply_thread_limit(n: int):
    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))


import h5py
from tqdm import tqdm

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.common.metrics import SSIM, foreground_mask, ssim_bbox, ssim_full
from utils.learning.test_part import load_model, prep_volume, recon_slice


def parse():
    parser = argparse.ArgumentParser(
        description='[CPU-only] Reconstruct + evaluate (SSIM_full / SSIM_bbox, no timing)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0,
                        help='Accepted for CLI compatibility but IGNORED (always uses CPU)')
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet')
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/')
    parser.add_argument('--cascade', type=int, default=1,
                        help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=9,
                        help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=4,
                        help='Number of channels for sensitivity map U-Net')
    parser.add_argument('--input_key', type=str, default='kspace')
    parser.add_argument('--num_threads', type=int, default=2,
                        help='Max CPU threads (keep low to avoid starving GPU training)')
    return parser.parse_args()


def run_acc(model, ssim, device, acc_dir):
    image_dir = acc_dir / 'image'
    kspace_dir = acc_dir / 'kspace'

    full_total, full_idx = 0.0, 0
    bbox_total, bbox_idx = 0.0, 0
    num_slices_total = 0

    volumes = sorted(image_dir.iterdir())
    acc_name = acc_dir.name  # 'acc4' or 'acc8'
    for img_path in tqdm(volumes, desc=f"[{acc_name}] volumes", unit="vol"):
        ks_path = kspace_dir / img_path.name
        with h5py.File(img_path, 'r') as hf:
            target_vol = hf['image_label'][:]
            maximum = hf.attrs['max']
            annotations = json.loads(hf.attrs.get('annotations', '{}'))

        ctx = prep_volume(img_path, ks_path if ks_path.exists() else None, device)
        n = ctx['num_slices']

        with torch.no_grad():
            recon_vol = []
            for s in range(n):
                out = recon_slice(model, ctx, s)
                recon_vol.append(out)

        num_slices_total += n

        for s in range(n):
            recon_t = recon_vol[s]
            target_t = torch.from_numpy(target_vol[s]).to(device=device)
            mask_t = torch.from_numpy(
                foreground_mask(target_vol[s])
            ).to(device=device).type(torch.float)

            value = ssim_full(ssim, recon_t, target_t, mask_t, maximum)
            if value is not None:
                full_total += value
                full_idx += 1
            for box in annotations.get(str(s), []):
                value = ssim_bbox(ssim, recon_t, target_t, box, maximum)
                if value is not None:
                    bbox_total += value
                    bbox_idx += 1

        del recon_vol

    full_score = full_total / full_idx if full_idx > 0 else 0.0
    bbox_score = bbox_total / bbox_idx if bbox_idx > 0 else 0.0
    return full_score, bbox_score, num_slices_total


if __name__ == '__main__':
    args = parse()
    _apply_thread_limit(args.num_threads)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    assert (args.path_data / 'acc4').is_dir() and (args.path_data / 'acc8').is_dir()

    device = torch.device('cpu')

    print("=" * 50)
    print("  recon_eval_cpuonly.py  (For running server)")
    print("  - Device: CPU (GPU not used)")
    print(f"  - Threads: {args.num_threads}")
    print("  - Timing: DISABLED (Recon Time = N/A)")
    print("  - Scoring: identical to recon_eval.py")
    print("=" * 50)

    model = load_model(args, device)
    ssim = SSIM().to(device=device)

    full4, bbox4, n4 = run_acc(model, ssim, device, args.path_data / 'acc4')
    full8, bbox8, n8 = run_acc(model, ssim, device, args.path_data / 'acc8')

    print()
    print("Leaderboard SSIM_full : {:.4f}".format((full4 + full8) / 2))
    print("Leaderboard SSIM_bbox : {:.4f}".format((bbox4 + bbox8) / 2))
    print("Leaderboard Recon Time : N/A (CPU-only mode, 리더보드 업로드 시 시간 항목은 공란)")
    print("=" * 10 + " Details " + "=" * 10)
    print("SSIM_full (acc4): {:.4f}   SSIM_full (acc8): {:.4f}".format(full4, full8))
    print("SSIM_bbox (acc4): {:.4f}   SSIM_bbox (acc8): {:.4f}".format(bbox4, bbox8))
    print("Recon Time: N/A")
    print("Total slices: {} (acc4: {}, acc8: {})".format(n4 + n8, n4, n8))