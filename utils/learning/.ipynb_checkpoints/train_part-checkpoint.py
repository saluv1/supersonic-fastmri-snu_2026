import os
import shutil
import time

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from utils.common.loss_function import (
    BBoxSSIMLoss,
    ForegroundSSIMLoss,
)
from utils.common.metrics import (
    SSIM,
    foreground_mask,
    ssim_bbox,
    ssim_full,
)
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.data.mask_augment import MaskAugmentor
from utils.data.mraugment.data_augment import DataAugmentor
from utils.model.promptmr_plus import PromptMR


def make_foreground_masks(target):
    """Build official foreground masks for a CPU target batch."""
    masks = np.stack(
        [
            foreground_mask(
                image.detach().cpu().numpy()
            )
            for image in target
        ]
    )

    return torch.from_numpy(masks).to(
        dtype=target.dtype
    )


def count_valid_boxes(
    boxes_batch,
    image_shape,
    win_size,
):
    """Count boxes large enough for valid-convolution SSIM."""
    height = int(image_shape[0])
    width = int(image_shape[1])

    per_sample = []

    for boxes in boxes_batch:
        count = 0

        for box in boxes:
            x0 = max(
                0,
                int(box["x"]),
            )
            y0 = max(
                0,
                int(box["y"]),
            )
            x1 = min(
                width,
                int(box["x"])
                + int(box["width"]),
            )
            y1 = min(
                height,
                int(box["y"])
                + int(box["height"]),
            )

            if (
                x1 - x0 >= win_size
                and y1 - y0 >= win_size
            ):
                count += 1

        per_sample.append(count)

    return per_sample


def acceleration_from_fname(fname):
    """Read the acceleration group from a challenge filename."""
    name = str(fname).lower()

    if "acc4" in name:
        return "acc4"

    if "acc8" in name:
        return "acc8"

    return "unknown"


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
        getattr(
            args,
            "bbox_loss_weight",
            0.3,
        )
    )

    total_loss = 0.0
    total_full_loss = 0.0
    total_bbox_loss = 0.0
    total_valid_boxes = 0
    total_annotated_slices = 0

    for iteration, data in enumerate(data_loader):
        (
            mask,
            kspace,
            target,
            maximum,
            _fnames,
            _slices,
            _bbox_mask,
            boxes_batch,
        ) = data

        foreground_masks = make_foreground_masks(
            target
        )

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
        foreground_masks = foreground_masks.to(
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
            foreground_masks,
        )

        valid_box_counts = count_valid_boxes(
            boxes_batch,
            image_shape=target.shape[-2:],
            win_size=bbox_loss_type.win_size,
        )

        current_box_count = sum(
            valid_box_counts
        )
        current_annotated_slices = sum(
            count > 0
            for count in valid_box_counts
        )

        if (
            bbox_weight > 0.0
            and current_box_count > 0
        ):
            bbox_loss = bbox_loss_type(
                output,
                target,
                maximum,
                boxes_batch,
            )
        else:
            bbox_loss = output.sum() * 0.0

        loss = (
            full_loss
            + bbox_weight * bbox_loss
        )

        optimizer.zero_grad(
            set_to_none=True
        )
        loss.backward()
        optimizer.step()

        total_loss += float(
            loss.item()
        )
        total_full_loss += float(
            full_loss.item()
        )
        total_bbox_loss += (
            float(bbox_loss.item())
            * current_box_count
        )
        total_valid_boxes += (
            current_box_count
        )
        total_annotated_slices += (
            current_annotated_slices
        )

        if iteration % args.report_interval == 0:
            print(
                f"Epoch = [{epoch:3d}/{args.num_epochs:3d}] "
                f"Iter = [{iteration:4d}/{len_loader:4d}] "
                f"Loss = {loss.item():.4g} "
                f"Full = {full_loss.item():.4g} "
                f"BBox = {bbox_loss.item():.4g} "
                f"Boxes = {current_box_count} "
                f"Time = "
                f"{time.perf_counter() - start_iter:.4f}s"
            )

            start_iter = time.perf_counter()

    loader_denominator = max(
        len_loader,
        1,
    )
    box_denominator = max(
        total_valid_boxes,
        1,
    )

    return (
        total_loss / loader_denominator,
        total_full_loss / loader_denominator,
        total_bbox_loss / box_denominator,
        total_annotated_slices,
        total_valid_boxes,
        time.perf_counter() - start_epoch,
    )


def validate(
    args,
    model,
    data_loader,
    metric_ssim,
    device,
):
    """Validate using the same metrics as recon_eval.py."""
    model.eval()

    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)

    start = time.perf_counter()

    full_totals = defaultdict(float)
    full_counts = defaultdict(int)

    bbox_totals = defaultdict(float)
    bbox_counts = defaultdict(int)

    seen_accelerations = set()
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
                _bbox_mask,
                boxes_batch,
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

            for index in range(
                output.shape[0]
            ):
                acceleration = (
                    acceleration_from_fname(
                        fnames[index]
                    )
                )
                seen_accelerations.add(
                    acceleration
                )

                reconstruction_t = output[index]
                target_t = target_device[index]

                mask_t = torch.from_numpy(
                    foreground_mask(
                        target[index]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                ).to(
                    device=device,
                    dtype=reconstruction_t.dtype,
                )

                full_value = ssim_full(
                    metric_ssim,
                    reconstruction_t,
                    target_t,
                    mask_t,
                    maximum[index],
                )

                if full_value is not None:
                    full_totals[
                        acceleration
                    ] += full_value
                    full_counts[
                        acceleration
                    ] += 1

                slice_has_valid_box = False

                for box in boxes_batch[index]:
                    bbox_value = ssim_bbox(
                        metric_ssim,
                        reconstruction_t,
                        target_t,
                        box,
                        maximum[index],
                    )

                    if bbox_value is not None:
                        bbox_totals[
                            acceleration
                        ] += bbox_value
                        bbox_counts[
                            acceleration
                        ] += 1

                        slice_has_valid_box = True

                if slice_has_valid_box:
                    annotated_slice_count += 1

                reconstructions[
                    fnames[index]
                ][
                    int(slices[index])
                ] = (
                    output[index]
                    .detach()
                    .cpu()
                    .numpy()
                )

                targets[
                    fnames[index]
                ][
                    int(slices[index])
                ] = (
                    target[index]
                    .detach()
                    .cpu()
                    .numpy()
                )

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [
                value
                for _, value in sorted(
                    reconstructions[
                        fname
                    ].items()
                )
            ]
        )

    for fname in targets:
        targets[fname] = np.stack(
            [
                value
                for _, value in sorted(
                    targets[
                        fname
                    ].items()
                )
            ]
        )

    ordered_accelerations = [
        acceleration
        for acceleration in (
            "acc4",
            "acc8",
            "unknown",
        )
        if acceleration in seen_accelerations
    ]

    full_by_acc = {}
    bbox_by_acc = {}

    for acceleration in ordered_accelerations:
        if full_counts[acceleration] > 0:
            full_by_acc[acceleration] = (
                full_totals[acceleration]
                / full_counts[acceleration]
            )
        else:
            full_by_acc[acceleration] = 0.0

        if bbox_counts[acceleration] > 0:
            bbox_by_acc[acceleration] = (
                bbox_totals[acceleration]
                / bbox_counts[acceleration]
            )
        else:
            bbox_by_acc[acceleration] = 0.0

    if full_by_acc:
        mean_full_score = float(
            np.mean(
                list(
                    full_by_acc.values()
                )
            )
        )
    else:
        mean_full_score = 0.0

    if bbox_by_acc:
        mean_bbox_score = float(
            np.mean(
                list(
                    bbox_by_acc.values()
                )
            )
        )
    else:
        mean_bbox_score = 0.0

    total_full_slices = sum(
        full_counts.values()
    )
    total_boxes = sum(
        bbox_counts.values()
    )

    return (
        mean_full_score,
        mean_bbox_score,
        total_full_slices,
        total_boxes,
        annotated_slice_count,
        full_by_acc,
        bbox_by_acc,
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
    scheduler,
    best_val_loss,
    is_new_best,
):
    """Save current training state and optionally update best_model.pt."""
    torch.save(
        {
            "epoch": epoch,
            "args": args,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
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
    """Verify PromptMR ACS auto-detection assumptions."""
    import h5py

    files = sorted(
        Path(
            data_path,
            "kspace",
        ).glob("*.h5")
    )[:n_files]

    if not files:
        raise FileNotFoundError(
            f"No .h5 found under "
            f"{Path(data_path, 'kspace')}"
        )

    for file_path in files:
        with h5py.File(
            file_path,
            "r",
        ) as hf:
            mask = np.asarray(
                hf["mask"]
            ).reshape(-1)

        width = len(mask)
        center = width // 2

        if mask[center] != 1:
            raise AssertionError(
                f"{file_path.name}: "
                "center line is not sampled; "
                "ACS auto-detection is invalid"
            )

        runs = np.where(
            np.diff(
                np.concatenate(
                    (
                        [0],
                        mask,
                        [0],
                    )
                )
            )
        )[0].reshape(-1, 2)

        start, end = next(
            (
                run_start,
                run_end,
            )
            for run_start, run_end in runs
            if run_start <= center < run_end
        )

        offset = abs(
            (start + end - 1) / 2
            - center
        )

        if offset > 2:
            raise AssertionError(
                f"{file_path.name}: "
                f"ACS center is off by "
                f"{offset:.1f} columns"
            )

    print(
        "[ACS] sanity check passed on "
        f"{len(files)} volumes"
    )


def build_model(args):
    """Single source of truth for training and inference."""
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
        torch.cuda.set_device(
            device
        )

        print(
            "Current cuda device:",
            torch.cuda.current_device(),
        )
    else:
        print(
            "CUDA unavailable; using CPU"
        )

    sanity_check_acs(
        args.data_path_train
    )

    model = build_model(
        args
    ).to(
        device=device
    )

    print(
        "Model params: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )
    print(
        "[Sensitivity] compute_sens_per_coil="
        f"{model.compute_sens_per_coil}"
    )

    full_loss_type = (
        ForegroundSSIMLoss()
        .to(device=device)
    )
    bbox_loss_type = (
        BBoxSSIMLoss()
        .to(device=device)
    )
    metric_ssim = SSIM().to(
        device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    scheduler = None

    if bool(
        getattr(
            args,
            "lr_scheduler",
            True,
        )
    ):
        lr_milestones = list(
            getattr(
                args,
                "lr_milestones",
                [16, 27],
            )
        )
        lr_gamma = float(
            getattr(
                args,
                "lr_gamma",
                0.3,
            )
        )

        scheduler = (
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=lr_milestones,
                gamma=lr_gamma,
            )
        )

        print(
            "[LR] MultiStepLR enabled | "
            f"initial={args.lr:g} "
            f"milestones={lr_milestones} "
            f"gamma={lr_gamma:g}"
        )
    else:
        print(
            "[LR] scheduler disabled | "
            f"lr={args.lr:g}"
        )

    bbox_weight = float(
        getattr(
            args,
            "bbox_loss_weight",
            0.3,
        )
    )

    print(
        "[Loss] foreground_ssim + "
        f"{bbox_weight:g} * bbox_ssim"
    )
    print(
        "[Validation] official foreground/bbox "
        "metrics with equal acc4/acc8 weighting"
    )

    best_val_loss = float("inf")
    start_epoch = 0

    current_epoch = [
        start_epoch
    ]

    mask_augmentor = None

    if args.mask_aug:
        mask_augmentor = MaskAugmentor(
            seed=args.seed,
            aug_weight=args.mask_aug_weight,
            aug_start=args.mask_aug_start,
            aug_schedule=(
                args.mask_aug_schedule
            ),
            aug_plateau_epoch=(
                args.mask_aug_plateau_epoch
            ),
            current_epoch_fn=(
                lambda: current_epoch[0]
            ),
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
            f"random_ratio="
            f"{args.mask_aug_random_ratio} "
            f"schedule="
            f"{args.mask_aug_schedule} "
            f"start={args.mask_aug_start} "
            f"plateau="
            f"{args.mask_aug_plateau_epoch}"
        )

    augmentor = None

    if args.aug_on:
        import argparse

        args_aug = argparse.Namespace(
            **vars(args)
        )

        args_aug.max_epochs = max(
            (
                args.num_epochs
                - args.annealing_epoch
            ),
            args.aug_delay + 1,
        )

        augmentor = DataAugmentor(
            args_aug,
            lambda: current_epoch[0],
            seed=args.seed,
        )

        augmentor.seed_pipeline(
            (
                args.seed + 2000
                if args.seed is not None
                else None
            )
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
        shuffle=False,
    )

    # Columns:
    #
    # epoch, train_total, train_foreground, train_bbox,
    # val_full_loss, val_bbox_loss, val_objective
    val_loss_log = np.empty(
        (0, 7)
    )

    for epoch in range(
        start_epoch,
        args.num_epochs,
    ):
        current_epoch[0] = epoch

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch #{epoch:2d} "
            f"............... "
            f"{args.net_name} "
            "............... "
            f"LR={current_lr:.6g}"
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
            train_annotated_slices,
            train_box_count,
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
            val_full_score,
            val_bbox_score,
            val_full_slice_count,
            val_box_count,
            val_annotated_slices,
            val_full_by_acc,
            val_bbox_by_acc,
            reconstructions,
            targets,
            inputs,
            val_time,
        ) = validate(
            args,
            model,
            val_loader,
            metric_ssim,
            device,
        )

        val_full_loss = (
            1.0
            - val_full_score
        )

        if val_box_count > 0:
            val_bbox_loss = (
                1.0
                - val_bbox_score
            )
        else:
            val_bbox_loss = 0.0

        val_objective = (
            val_full_loss
            + bbox_weight
            * val_bbox_loss
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

        np.save(
            file_path,
            val_loss_log,
        )

        is_new_best = (
            val_objective
            < best_val_loss
        )
        best_val_loss = min(
            best_val_loss,
            val_objective,
        )

        # Step after finishing the current epoch. Therefore milestone 16
        # means epoch 16 starts with the reduced learning rate.
        if scheduler is not None:
            scheduler.step()

        next_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        save_model(
            args,
            args.exp_dir,
            epoch + 1,
            model,
            optimizer,
            scheduler,
            best_val_loss,
            is_new_best,
        )

        print(
            f"Epoch = "
            f"[{epoch:4d}/{args.num_epochs:4d}] "
            f"Train = {train_loss:.4g} "
            f"TrainFull = {train_full_loss:.4g} "
            f"TrainBBox = {train_bbox_loss:.4g} "
            f"ValFull = {val_full_loss:.4g} "
            f"ValBBox = {val_bbox_loss:.4g} "
            f"ValObjective = {val_objective:.4g} "
            f"ValSSIMFull = {val_full_score:.4f} "
            f"ValSSIMBBox = {val_bbox_score:.4f} "
            f"TrainAnnotatedSlices = "
            f"{train_annotated_slices} "
            f"TrainBoxes = {train_box_count} "
            f"ValAnnotatedSlices = "
            f"{val_annotated_slices} "
            f"ValBoxes = {val_box_count} "
            f"ValFullSlices = "
            f"{val_full_slice_count} "
            f"LR = {current_lr:.6g} "
            f"NextLR = {next_lr:.6g} "
            f"TrainTime = {train_time:.4f}s "
            f"ValTime = {val_time:.4f}s"
        )

        print(
            "  Official validation by acceleration | "
            f"full={val_full_by_acc} "
            f"bbox={val_bbox_by_acc}"
        )

        if is_new_best:
            print(
                "@@@@@@@@@@@@@@@@@@@@"
                "NewRecord"
                "@@@@@@@@@@@@@@@@@@@@"
            )

            save_start = time.perf_counter()

            save_reconstructions(
                reconstructions,
                args.val_dir,
                targets=targets,
                inputs=inputs,
            )

            print(
                "ForwardTime = "
                f"{time.perf_counter() - save_start:.4f}s"
            )