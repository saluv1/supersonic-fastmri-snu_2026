import os
import shutil
import time
from collections import defaultdict

import numpy as np
import torch

from utils.common.loss_function import BBoxSSIMLoss, SSIMLoss
from utils.common.utils import save_reconstructions, ssim_loss
from utils.data.load_data import create_data_loaders
from utils.data.mask_augment import MaskAugmentor
from utils.data.mraugment.data_augment import DataAugmentor
from utils.model.promptmr_plus import PromptMR


def train_epoch(
    args,
    epoch,
    model,
    data_loader,
    optimizer,
    full_loss_type,
    bbox_loss_type,
    device,
):
    model.train()

    start_epoch = time.perf_counter()
    start_iter = start_epoch
    len_loader = len(data_loader)

    bbox_weight = float(
        getattr(args, "bbox_loss_weight", 0.3)
    )

    total_loss = 0.0
    total_full_loss = 0.0
    total_bbox_loss = 0.0

    for iteration, data in enumerate(data_loader):
        (
            mask,
            kspace,
            target,
            maximum,
            _,
            _,
            bbox_mask,
        ) = data

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

        output = model(
            kspace,
            mask,
            use_checkpoint=args.use_checkpoint,
            compute_sens_per_coil=(
                model.compute_sens_per_coil
            ),
        )

        full_loss = full_loss_type(
            output,
            target,
            maximum,
        )

        if bbox_weight > 0.0:
            bbox_loss = bbox_loss_type(
                output,
                target,
                maximum,
                bbox_mask,
            )
        else:
            bbox_loss = output.sum() * 0.0

        loss = (
            full_loss
            + bbox_weight * bbox_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_full_loss += float(full_loss.item())
        total_bbox_loss += float(bbox_loss.item())

        if iteration % args.report_interval == 0:
            print(
                f"Epoch = [{epoch:3d}/{args.num_epochs:3d}] "
                f"Iter = [{iteration:4d}/{len_loader:4d}] "
                f"Loss = {loss.item():.4g} "
                f"Full = {full_loss.item():.4g} "
                f"BBox = {bbox_loss.item():.4g} "
                f"Time = {time.perf_counter() - start_iter:.4f}s"
            )
            start_iter = time.perf_counter()

    denominator = max(len_loader, 1)

    return (
        total_loss / denominator,
        total_full_loss / denominator,
        total_bbox_loss / denominator,
        time.perf_counter() - start_epoch,
    )


def validate(
    args,
    model,
    data_loader,
    bbox_loss_type,
    device,
):
    model.eval()

    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    bbox_loss_total = 0.0
    annotated_slice_count = 0

    with torch.no_grad():
        for data in data_loader:
            (
                mask,
                kspace,
                target,
                maximum,
                fnames,
                slices,
                bbox_mask,
            ) = data

            kspace = kspace.to(
                device=device,
                non_blocking=True,
            )
            mask = mask.to(
                device=device,
                non_blocking=True,
            )

            target_device = target.to(
                device=device,
                non_blocking=True,
            )
            bbox_mask_device = bbox_mask.to(
                device=device,
                non_blocking=True,
            )
            maximum = maximum.to(
                device=device,
                non_blocking=True,
            )

            output = model(
                kspace,
                mask,
                use_checkpoint=False,
                compute_sens_per_coil=(
                    model.compute_sens_per_coil
                ),
            )

            has_box = (
                bbox_mask_device
                .flatten(start_dim=1)
                .sum(dim=1)
                > 0
            )
            current_annotated = int(
                has_box.sum().item()
            )

            if current_annotated > 0:
                current_bbox_loss = bbox_loss_type(
                    output,
                    target_device,
                    maximum,
                    bbox_mask_device,
                )
                bbox_loss_total += (
                    float(current_bbox_loss.item())
                    * current_annotated
                )
                annotated_slice_count += current_annotated

            for index in range(output.shape[0]):
                reconstructions[fnames[index]][
                    int(slices[index])
                ] = output[index].cpu().numpy()

                targets[fnames[index]][
                    int(slices[index])
                ] = target[index].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [
                output
                for _, output in sorted(
                    reconstructions[fname].items()
                )
            ]
        )

    for fname in targets:
        targets[fname] = np.stack(
            [
                target
                for _, target in sorted(
                    targets[fname].items()
                )
            ]
        )

    metric_loss = sum(
        ssim_loss(
            targets[fname],
            reconstructions[fname],
        )
        for fname in reconstructions
    )

    num_subjects = len(reconstructions)

    if annotated_slice_count > 0:
        mean_bbox_loss = (
            bbox_loss_total
            / annotated_slice_count
        )
    else:
        mean_bbox_loss = 0.0

    return (
        metric_loss,
        num_subjects,
        mean_bbox_loss,
        annotated_slice_count,
        reconstructions,
        targets,
        None,
        time.perf_counter() - start,
    )


def save_model(
    args,
    exp_dir,
    epoch,
    model,
    optimizer,
    best_val_loss,
    is_new_best,
):
    torch.save(
        {
            "epoch": epoch,
            "args": args,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "exp_dir": exp_dir,
        },
        f=exp_dir / "model.pt",
    )

    if is_new_best:
        shutil.copyfile(
            exp_dir / "model.pt",
            exp_dir / "best_model.pt",
        )


def sanity_check_acs(
    data_path,
    n_files=8,
):
    """Verify assumptions required by PromptMR ACS auto-detection."""
    import h5py
    from pathlib import Path

    files = sorted(
        Path(data_path, "kspace").glob("*.h5")
    )[:n_files]

    if not files:
        raise FileNotFoundError(
            f"No .h5 found under "
            f"{Path(data_path, 'kspace')}"
        )

    for file_path in files:
        with h5py.File(file_path, "r") as hf:
            mask = np.asarray(
                hf["mask"]
            ).reshape(-1)

        width = len(mask)
        center = width // 2

        if mask[center] != 1:
            raise AssertionError(
                f"{file_path.name}: center line is not sampled; "
                "ACS auto-detection is invalid"
            )

        runs = np.where(
            np.diff(
                np.concatenate(
                    ([0], mask, [0])
                )
            )
        )[0].reshape(-1, 2)

        start, end = next(
            (run_start, run_end)
            for run_start, run_end in runs
            if run_start <= center < run_end
        )

        offset = abs(
            (start + end - 1) / 2
            - center
        )

        if offset > 2:
            raise AssertionError(
                f"{file_path.name}: ACS center is off by "
                f"{offset:.1f} columns"
            )

    print(
        f"[ACS] sanity check passed on "
        f"{len(files)} volumes"
    )


def build_model(args):
    """Single source of truth for train and inference construction."""
    model = PromptMR(
        num_cascades=args.num_cascades,
        num_adj_slices=1,
        n_feat0=args.n_feat0,
        feature_dim=args.feature_dim,
        prompt_dim=args.prompt_dim,
        sens_n_feat0=args.sens_n_feat0,
        sens_feature_dim=args.sens_feature_dim,
        sens_prompt_dim=args.sens_prompt_dim,
        len_prompt=args.len_prompt,
        prompt_size=args.prompt_size,
        n_enc_cab=args.n_enc_cab,
        n_dec_cab=args.n_dec_cab,
        n_skip_cab=args.n_skip_cab,
        n_bottleneck_cab=args.n_bottleneck_cab,
        no_use_ca=args.no_use_ca,
        adaptive_input=args.adaptive_input,
        n_buffer=args.n_buffer,
        n_history=args.n_history,
        use_sens_adj=args.use_sens_adj,
    )
    # This is a forward-execution option, not a model parameter. Keeping it
    # on the model makes the checkpoint's training choice available to the
    # fixed inference harness without changing recon_eval.py.
    model.compute_sens_per_coil = bool(
        getattr(
            args,
            "compute_sens_per_coil",
            True,
        )
    )
    return model


def train(args):
    device = torch.device(
        f"cuda:{args.GPU_NUM}"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(
            "Current cuda device:",
            torch.cuda.current_device(),
        )
    else:
        print("CUDA unavailable; using CPU")

    sanity_check_acs(args.data_path_train)

    model = build_model(args).to(device=device)

    print(
        "Model params: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )
    print(
        "[Sensitivity] compute_sens_per_coil="
        f"{model.compute_sens_per_coil}"
    )

    full_loss_type = SSIMLoss().to(device=device)
    bbox_loss_type = BBoxSSIMLoss().to(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    bbox_weight = float(
        getattr(args, "bbox_loss_weight", 0.3)
    )
    print(
        f"[Loss] full_ssim + "
        f"{bbox_weight:g} * bbox_ssim"
    )

    best_val_loss = float("inf")
    start_epoch = 0

    current_epoch = [start_epoch]

    mask_augmentor = None
    if args.mask_aug:
        mask_augmentor = MaskAugmentor(
            seed=args.seed,
            aug_weight=args.mask_aug_weight,
            aug_start=args.mask_aug_start,
            aug_schedule=args.mask_aug_schedule,
            aug_plateau_epoch=(
                args.mask_aug_plateau_epoch
            ),
            current_epoch_fn=lambda: current_epoch[0],
            accelerations=(
                args.mask_aug_accelerations
            ),
            random_ratio=(
                args.mask_aug_random_ratio
            ),
            random_offset=(
                args.mask_aug_random_offset
            ),
        )

        print(
            "[MaskAug] enabled | "
            f"acc={args.mask_aug_accelerations} "
            f"random_ratio={args.mask_aug_random_ratio} "
            f"schedule={args.mask_aug_schedule} "
            f"start={args.mask_aug_start} "
            f"plateau={args.mask_aug_plateau_epoch}"
        )

    augmentor = None
    if args.aug_on:
        import argparse

        args_aug = argparse.Namespace(
            **vars(args)
        )
        args_aug.max_epochs = max(
            args.num_epochs - args.annealing_epoch,
            args.aug_delay + 1,
        )

        augmentor = DataAugmentor(
            args_aug,
            lambda: current_epoch[0],
            seed=args.seed,
        )
        augmentor.seed_pipeline(
            args.seed + 2000
            if args.seed is not None
            else None
        )

        print(
            "[MRAugment] enabled | "
            f"schedule={args.aug_schedule} "
            f"delay={args.aug_delay} "
            f"strength={args.aug_strength} "
            f"max_epochs={args_aug.max_epochs}"
        )

    train_loader = create_data_loaders(
        data_path=args.data_path_train,
        args=args,
        shuffle=True,
        mask_augmentor=mask_augmentor,
        augmentor=augmentor,
    )

    val_loader = create_data_loaders(
        data_path=args.data_path_val,
        args=args,
    )

    # Columns:
    # epoch, train_total, train_full, train_bbox,
    # val_full, val_bbox, val_objective
    val_loss_log = np.empty((0, 7))

    for epoch in range(
        start_epoch,
        args.num_epochs,
    ):
        current_epoch[0] = epoch

        print(
            f"Epoch #{epoch:2d} "
            f"............... {args.net_name} "
            "..............."
        )

        if mask_augmentor is not None:
            print(
                "  mask-aug p = "
                f"{mask_augmentor.schedule_p():.3f}"
            )

        if augmentor is not None:
            print(
                "  mr-aug   p = "
                f"{augmentor.schedule_p():.3f}"
            )

        (
            train_loss,
            train_full_loss,
            train_bbox_loss,
            train_time,
        ) = train_epoch(
            args,
            epoch,
            model,
            train_loader,
            optimizer,
            full_loss_type,
            bbox_loss_type,
            device,
        )

        (
            val_metric_sum,
            num_subjects,
            val_bbox_loss,
            annotated_slice_count,
            reconstructions,
            targets,
            inputs,
            val_time,
        ) = validate(
            args,
            model,
            val_loader,
            bbox_loss_type,
            device,
        )

        val_full_loss = (
            val_metric_sum
            / max(num_subjects, 1)
        )
        val_objective = (
            val_full_loss
            + bbox_weight * val_bbox_loss
        )

        row = np.array(
            [
                [
                    epoch,
                    train_loss,
                    train_full_loss,
                    train_bbox_loss,
                    val_full_loss,
                    val_bbox_loss,
                    val_objective,
                ]
            ]
        )
        val_loss_log = np.append(
            val_loss_log,
            row,
            axis=0,
        )

        file_path = os.path.join(
            args.val_loss_dir,
            "val_loss_log",
        )
        np.save(file_path, val_loss_log)

        is_new_best = (
            val_objective < best_val_loss
        )
        best_val_loss = min(
            best_val_loss,
            val_objective,
        )

        save_model(
            args,
            args.exp_dir,
            epoch + 1,
            model,
            optimizer,
            best_val_loss,
            is_new_best,
        )

        print(
            f"Epoch = [{epoch:4d}/{args.num_epochs:4d}] "
            f"Train = {train_loss:.4g} "
            f"TrainFull = {train_full_loss:.4g} "
            f"TrainBBox = {train_bbox_loss:.4g} "
            f"ValFull = {val_full_loss:.4g} "
            f"ValBBox = {val_bbox_loss:.4g} "
            f"ValObjective = {val_objective:.4g} "
            f"AnnotatedSlices = {annotated_slice_count} "
            f"TrainTime = {train_time:.4f}s "
            f"ValTime = {val_time:.4f}s"
        )

        if is_new_best:
            print(
                "@@@@@@@@@@@@@@@@@@@@"
                "NewRecord"
                "@@@@@@@@@@@@@@@@@@@@"
            )

            start = time.perf_counter()
            save_reconstructions(
                reconstructions,
                args.val_dir,
                targets=targets,
                inputs=inputs,
            )
            print(
                "ForwardTime = "
                f"{time.perf_counter() - start:.4f}s"
            )
