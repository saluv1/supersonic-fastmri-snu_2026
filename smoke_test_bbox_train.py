"""Run one PromptMR+ training step on an annotated 2026 slice.

This script does not save a checkpoint or reconstruction. It verifies:

1. annotation loading and bbox-mask construction,
2. PromptMR+ forward,
3. full-image and bbox SSIM losses,
4. backward gradients,
5. one optimizer step.

Run this from the FastMRI_challenge repository root.
"""

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "utils" / "model"

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.common.loss_function import BBoxSSIMLoss, SSIMLoss
from utils.data.load_data import create_data_loaders, get_slice_boxes
from utils.learning.train_part import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one forward/backward/optimizer step on an annotated slice."
        )
    )
    parser.add_argument(
        "-g",
        "--gpu-num",
        type=int,
        default=0,
        help="CUDA device number",
    )
    parser.add_argument(
        "-t",
        "--data-path",
        type=Path,
        default=Path("/root/Data/train/"),
        help="2026 training-data directory",
    )
    parser.add_argument(
        "--bbox-loss-weight",
        type=float,
        default=0.3,
        help="Weight applied to BBoxSSIMLoss",
    )
    parser.add_argument(
        "--num-cascades",
        type=int,
        default=1,
        help="Use a small model for the smoke test",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    return parser.parse_args()


def make_model_args(cli_args):
    """Use the lightweight PromptMR+ defaults currently in train.py."""
    return SimpleNamespace(
        num_cascades=cli_args.num_cascades,
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
        use_checkpoint=not cli_args.no_checkpoint,
    )


def make_data_args():
    """Arguments consumed by create_data_loaders."""
    return SimpleNamespace(
        input_key="kspace",
        target_key="image_label",
        max_key="max",
        batch_size=1,
    )


def find_annotated_index(dataset):
    """Find a dataset index without transforming every preceding slice."""
    cached_path = None
    cached_attrs = None

    for index, (image_path, slice_index) in enumerate(
        dataset.image_examples
    ):
        if image_path != cached_path:
            with h5py.File(image_path, "r") as hf:
                cached_attrs = dict(hf.attrs)
            cached_path = image_path

        boxes = get_slice_boxes(
            cached_attrs,
            slice_index,
        )
        if boxes:
            return index, image_path, slice_index, boxes

    raise RuntimeError(
        "No annotated slice was found in the training dataset."
    )


def main():
    cli_args = parse_args()

    if cli_args.bbox_loss_weight < 0:
        raise ValueError(
            "--bbox-loss-weight must be greater than or equal to zero"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this PromptMR+ smoke test."
        )

    device = torch.device(f"cuda:{cli_args.gpu_num}")
    torch.cuda.set_device(device)
    torch.manual_seed(430)
    torch.cuda.manual_seed_all(430)

    data_args = make_data_args()
    loader = create_data_loaders(
        data_path=cli_args.data_path,
        args=data_args,
        shuffle=False,
    )
    dataset = loader.dataset

    (
        annotated_index,
        image_path,
        slice_index,
        boxes,
    ) = find_annotated_index(dataset)

    one_slice_loader = DataLoader(
        Subset(dataset, [annotated_index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(one_slice_loader))
    if len(batch) != 7:
        raise RuntimeError(
            f"Expected 7 batch items, received {len(batch)}."
        )

    (
        mask,
        kspace,
        target,
        maximum,
        _,
        _,
        bbox_mask,
    ) = batch

    mask = mask.to(
        device=device,
        non_blocking=True,
    )
    kspace = kspace.to(
        device=device,
        non_blocking=True,
    )
    target = target.to(
        device=device,
        non_blocking=True,
    )
    maximum = maximum.to(
        device=device,
        non_blocking=True,
    )
    bbox_mask = bbox_mask.to(
        device=device,
        non_blocking=True,
    )

    if bbox_mask.sum().item() <= 0:
        raise RuntimeError(
            "The selected annotated slice produced an empty bbox mask."
        )

    model_args = make_model_args(cli_args)
    model = build_model(model_args).to(device=device)
    model.train()

    full_loss_fn = SSIMLoss().to(device=device)
    bbox_loss_fn = BBoxSSIMLoss().to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.01,
    )

    optimizer.zero_grad(set_to_none=True)

    output = model(
        kspace,
        mask,
        use_checkpoint=model_args.use_checkpoint,
    )

    if output.shape != target.shape:
        raise RuntimeError(
            f"Output/target shape mismatch: "
            f"{tuple(output.shape)} vs {tuple(target.shape)}"
        )

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "Model output contains NaN or Inf."
        )

    full_loss = full_loss_fn(
        output,
        target,
        maximum,
    )
    bbox_loss = bbox_loss_fn(
        output,
        target,
        maximum,
        bbox_mask,
    )
    total_loss = (
        full_loss
        + cli_args.bbox_loss_weight * bbox_loss
    )

    for loss_name, loss_value in (
        ("full_loss", full_loss),
        ("bbox_loss", bbox_loss),
        ("total_loss", total_loss),
    ):
        if not torch.isfinite(loss_value):
            raise RuntimeError(
                f"{loss_name} is NaN or Inf."
            )

    total_loss.backward()

    gradient_square_sum = 0.0
    gradient_tensor_count = 0

    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(
                f"Gradient contains NaN or Inf: {parameter_name}"
            )

        gradient_square_sum += (
            parameter.grad.detach().float().square().sum().item()
        )
        gradient_tensor_count += 1

    gradient_norm = math.sqrt(gradient_square_sum)

    if gradient_tensor_count == 0 or gradient_norm <= 0:
        raise RuntimeError(
            "No non-zero model gradient was produced."
        )

    optimizer.step()
    torch.cuda.synchronize(device)

    print("PromptMR bbox training-step smoke test OK")
    print(f"device: {device}")
    print(f"file: {image_path.name}")
    print(f"slice: {slice_index}")
    print(f"boxes: {boxes}")
    print(f"output: {tuple(output.shape)}")
    print(f"bbox pixels: {int(bbox_mask.sum().item())}")
    print(f"full loss: {full_loss.item():.6f}")
    print(f"bbox loss: {bbox_loss.item():.6f}")
    print(f"total loss: {total_loss.item():.6f}")
    print(f"gradient norm: {gradient_norm:.6f}")
    print(f"gradient tensors: {gradient_tensor_count}")
    print(
        "CUDA peak allocated: "
        f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB"
    )
    print(
        "CUDA peak reserved: "
        f"{torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GiB"
    )
    print("No checkpoint or reconstruction was written.")


if __name__ == "__main__":
    main()
