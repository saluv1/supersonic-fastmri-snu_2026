import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "utils" / "model"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


from utils.common.loss_function import (
    BBoxSSIMLoss,
    ForegroundSSIMLoss,
)
from utils.common.metrics import (
    SSIM,
    ssim_bbox,
    ssim_full,
)
from utils.data.load_data import create_data_loaders
from utils.learning.train_part import (
    build_model,
    make_foreground_masks,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-t",
        "--data-path",
        type=Path,
        default=ROOT.parent / "Data" / "train",
    )
    parser.add_argument(
        "--bbox-loss-weight",
        type=float,
        default=0.3,
    )

    return parser.parse_args()


def model_args():
    return argparse.Namespace(
        num_cascades=1,
        n_feat0=8,
        feature_dim=[24, 32, 40],
        prompt_dim=[8, 16, 24],
        sens_n_feat0=8,
        sens_feature_dim=[12, 16, 20],
        sens_prompt_dim=[4, 8, 12],
        len_prompt=[3, 3, 3],
        prompt_size=[16, 8, 4],
        n_enc_cab=[2, 3, 3],
        n_dec_cab=[2, 2, 3],
        n_skip_cab=[1, 1, 1],
        n_bottleneck_cab=3,
        no_use_ca=False,
        adaptive_input=True,
        n_buffer=4,
        n_history=0,
        use_sens_adj=True,
        compute_sens_per_coil=True,
    )


def loader_args():
    return argparse.Namespace(
        batch_size=1,
        input_key="kspace",
        target_key="image_label",
        max_key="max",
    )


def find_annotated_batch(loader):
    for batch in loader:
        if len(batch) != 8:
            raise AssertionError(
                f"Expected 8 fields, got {len(batch)}"
            )

        boxes_batch = batch[7]

        if any(
            len(boxes) > 0
            for boxes in boxes_batch
        ):
            return batch

    raise RuntimeError(
        "No annotated slice was found in the dataset"
    )


def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    loader = create_data_loaders(
        data_path=args.data_path,
        args=loader_args(),
        shuffle=False,
        isforward=False,
    )

    batch = find_annotated_batch(loader)

    (
        mask,
        kspace,
        target,
        maximum,
        fnames,
        slices,
        bbox_mask,
        boxes_batch,
    ) = batch

    foreground_masks = make_foreground_masks(
        target
    )

    mask = mask.to(device=device)
    kspace = kspace.to(device=device)
    target = target.to(device=device)
    maximum = maximum.to(device=device)
    foreground_masks = foreground_masks.to(
        device=device
    )

    model = build_model(
        model_args()
    ).to(
        device=device
    )
    model.train()

    output = model(
        kspace,
        mask,
        use_checkpoint=False,
        compute_sens_per_coil=True,
    )

    full_loss_fn = ForegroundSSIMLoss().to(
        device=device
    )
    bbox_loss_fn = BBoxSSIMLoss().to(
        device=device
    )

    full_loss = full_loss_fn(
        output,
        target,
        maximum,
        foreground_masks,
    )

    bbox_loss = bbox_loss_fn(
        output,
        target,
        maximum,
        boxes_batch,
    )

    total_loss = (
        full_loss
        + args.bbox_loss_weight * bbox_loss
    )

    if not torch.isfinite(total_loss):
        raise AssertionError(
            f"Non-finite total loss: {total_loss.item()}"
        )

    total_loss.backward()

    gradient_tensors = 0
    gradient_norm_squared = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        if not torch.isfinite(
            parameter.grad
        ).all():
            raise AssertionError(
                "A non-finite model gradient was found"
            )

        gradient_tensors += 1

        gradient_norm_squared += float(
            parameter.grad
            .detach()
            .float()
            .norm()
            .item() ** 2
        )

    if gradient_tensors == 0:
        raise AssertionError(
            "No model gradients were produced"
        )

    metric = SSIM().to(
        device=device
    )

    with torch.no_grad():
        official_full_score = ssim_full(
            metric,
            output[0].detach(),
            target[0],
            foreground_masks[0],
            maximum[0],
        )

        official_bbox_scores = []

        for box in boxes_batch[0]:
            value = ssim_bbox(
                metric,
                output[0].detach(),
                target[0],
                box,
                maximum[0],
            )

            if value is not None:
                official_bbox_scores.append(
                    value
                )

    if official_full_score is None:
        raise AssertionError(
            "The foreground mask was unexpectedly empty"
        )

    if not official_bbox_scores:
        raise AssertionError(
            "No valid bbox score was produced"
        )

    expected_full_loss = (
        1.0 - official_full_score
    )
    expected_bbox_loss = (
        1.0
        - float(
            np.mean(
                official_bbox_scores
            )
        )
    )

    full_error = abs(
        float(full_loss.item())
        - expected_full_loss
    )
    bbox_error = abs(
        float(bbox_loss.item())
        - expected_bbox_loss
    )

    if full_error > 1e-5:
        raise AssertionError(
            "Foreground loss does not match metrics.py: "
            f"error={full_error:.8g}"
        )

    if bbox_error > 1e-5:
        raise AssertionError(
            "BBox loss does not match metrics.py: "
            f"error={bbox_error:.8g}"
        )

    print(
        "Metric-aligned PromptMR training smoke test OK"
    )
    print(
        "device:",
        device,
    )
    print(
        "file:",
        fnames[0],
    )
    print(
        "slice:",
        int(slices[0]),
    )
    print(
        "boxes:",
        boxes_batch[0],
    )
    print(
        "output:",
        tuple(output.shape),
    )
    print(
        "legacy bbox-mask pixels:",
        int(bbox_mask.sum().item()),
    )
    print(
        "foreground loss:",
        f"{full_loss.item():.6f}",
    )
    print(
        "official full loss:",
        f"{expected_full_loss:.6f}",
    )
    print(
        "full alignment error:",
        f"{full_error:.3e}",
    )
    print(
        "per-box bbox loss:",
        f"{bbox_loss.item():.6f}",
    )
    print(
        "official bbox loss:",
        f"{expected_bbox_loss:.6f}",
    )
    print(
        "bbox alignment error:",
        f"{bbox_error:.3e}",
    )
    print(
        "total loss:",
        f"{total_loss.item():.6f}",
    )
    print(
        "gradient norm:",
        f"{gradient_norm_squared ** 0.5:.6f}",
    )
    print(
        "gradient tensors:",
        gradient_tensors,
    )

    if device.type == "cuda":
        print(
            "CUDA peak allocated:",
            f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB",
        )
        print(
            "CUDA peak reserved:",
            f"{torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB",
        )

    print(
        "No optimizer step, checkpoint, "
        "or reconstruction was written."
    )


if __name__ == "__main__":
    main()
