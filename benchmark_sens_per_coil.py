"""Benchmark PromptMR sensitivity-map execution modes on one volume.

The same checkpoint and the same slices are evaluated with
compute_sens_per_coil=True and False. No checkpoint or reconstruction is
written. If the parallel mode runs out of memory, the script reports OOM and
keeps True as the safe choice.
"""

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "utils" / "model"

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.learning.test_part import (
    load_model,
    prep_volume,
    recon_slice,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare PromptMR compute_sens_per_coil=True/False "
            "using one checkpoint and volume."
        )
    )
    parser.add_argument(
        "-g",
        "--gpu-num",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-n",
        "--net-name",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--acc-dir",
        type=Path,
        required=True,
        help="Directory containing image/ and kspace/",
    )
    parser.add_argument(
        "--warmup-slices",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=0,
        help="0 uses every slice in the selected volume.",
    )
    return parser.parse_args()


def percentile_95(values):
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(0.95 * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def benchmark_mode(
    model,
    ctx,
    slice_indices,
    warmup_slices,
    mode,
    device,
):
    model.compute_sens_per_coil = mode

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    outputs = []
    elapsed = []

    with torch.no_grad():
        for slice_index in slice_indices[:warmup_slices]:
            recon_slice(
                model,
                ctx,
                slice_index,
            )

        torch.cuda.synchronize(device)

        for slice_index in slice_indices:
            torch.cuda.synchronize(device)
            start = time.perf_counter()

            output = recon_slice(
                model,
                ctx,
                slice_index,
            )

            torch.cuda.synchronize(device)
            elapsed.append(
                time.perf_counter() - start
            )
            outputs.append(
                output.detach().cpu()
            )

    return {
        "mode": mode,
        "outputs": torch.stack(outputs),
        "mean_ms": statistics.mean(elapsed) * 1000,
        "median_ms": statistics.median(elapsed) * 1000,
        "p95_ms": percentile_95(elapsed) * 1000,
        "total_s": sum(elapsed),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device)
            / 1024**3
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device)
            / 1024**3
        ),
    }


def print_result(result):
    print(
        "compute_sens_per_coil="
        f"{result['mode']}"
    )
    print(
        f"  mean:   {result['mean_ms']:.1f} ms/slice"
    )
    print(
        f"  median: {result['median_ms']:.1f} ms/slice"
    )
    print(
        f"  p95:    {result['p95_ms']:.1f} ms/slice"
    )
    print(
        f"  total:  {result['total_s']:.2f} s"
    )
    print(
        "  peak allocated: "
        f"{result['peak_allocated_gib']:.2f} GiB"
    )
    print(
        "  peak reserved:  "
        f"{result['peak_reserved_gib']:.2f} GiB"
    )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this benchmark."
        )

    device = torch.device(
        f"cuda:{args.gpu_num}"
    )
    torch.cuda.set_device(device)

    image_files = sorted(
        (args.acc_dir / "image").glob("*.h5")
    )
    if not image_files:
        raise FileNotFoundError(
            f"No image H5 found in {args.acc_dir / 'image'}"
        )

    image_path = image_files[0]
    kspace_path = (
        args.acc_dir
        / "kspace"
        / image_path.name
    )
    if not kspace_path.is_file():
        raise FileNotFoundError(
            f"Missing matching k-space H5: {kspace_path}"
        )

    checkpoint_args = SimpleNamespace(
        exp_dir=(
            PROJECT_ROOT.parent
            / "result"
            / args.net_name
            / "checkpoints"
        )
    )
    model = load_model(
        checkpoint_args,
        device,
    )
    ctx = prep_volume(
        image_path,
        kspace_path,
        device,
    )

    num_slices = ctx["num_slices"]
    if args.max_slices > 0:
        num_slices = min(
            num_slices,
            args.max_slices,
        )
    slice_indices = list(range(num_slices))

    print(
        "GPU:",
        torch.cuda.get_device_name(device),
    )
    print("volume:", image_path.name)
    print("timed slices:", len(slice_indices))
    print(
        "checkpoint:",
        checkpoint_args.exp_dir / "best_model.pt",
    )

    results = {}

    for mode in (True, False):
        try:
            result = benchmark_mode(
                model=model,
                ctx=ctx,
                slice_indices=slice_indices,
                warmup_slices=min(
                    args.warmup_slices,
                    len(slice_indices),
                ),
                mode=mode,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            print(
                "compute_sens_per_coil="
                f"{mode}: CUDA OOM"
            )
            print(
                "Use compute_sens_per_coil=True "
                "on this GPU."
            )
            torch.cuda.empty_cache()
            continue

        results[mode] = result
        print_result(result)

    if True in results and False in results:
        true_result = results[True]
        false_result = results[False]

        difference = (
            true_result["outputs"]
            - false_result["outputs"]
        ).abs()

        speedup = (
            true_result["mean_ms"]
            / false_result["mean_ms"]
        )
        extra_memory = (
            false_result["peak_allocated_gib"]
            - true_result["peak_allocated_gib"]
        )

        print("========== comparison ==========")
        print(
            f"False speedup: {speedup:.3f}x"
        )
        print(
            "False extra peak allocation: "
            f"{extra_memory:+.2f} GiB"
        )
        print(
            "output mean absolute difference: "
            f"{difference.mean().item():.8g}"
        )
        print(
            "output max absolute difference: "
            f"{difference.max().item():.8g}"
        )

        if false_result["mean_ms"] < true_result["mean_ms"]:
            print(
                "Faster mode on this GPU: "
                "compute_sens_per_coil=False"
            )
        else:
            print(
                "Faster mode on this GPU: "
                "compute_sens_per_coil=True"
            )

    print(
        "Benchmark complete; no checkpoint or "
        "reconstruction was written."
    )


if __name__ == "__main__":
    main()
