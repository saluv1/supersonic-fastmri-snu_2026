"""SSIM losses used for PromptMR training.

SSIMLoss:
    Original whole-image SSIM loss.

ForegroundSSIMLoss:
    Differentiable loss corresponding to the leaderboard SSIM_full metric.

BBoxSSIMLoss:
    Computes SSIM loss separately for every annotation box and gives every
    valid box equal weight, matching the leaderboard SSIM_bbox averaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """Original full-image SSIM loss."""

    def __init__(
        self,
        win_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()

        self.win_size = win_size
        self.k1 = k1
        self.k2 = k2

        self.register_buffer(
            "w",
            torch.ones(
                1,
                1,
                win_size,
                win_size,
            ) / win_size**2,
        )

        num_pixels = win_size**2
        self.cov_norm = (
            num_pixels
            / (num_pixels - 1)
        )

    def _ssim_map(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
    ) -> torch.Tensor:
        """Return a valid-convolution SSIM map.

        Args:
            X:
                Reconstruction with shape (B,H,W).

            Y:
                Target with shape (B,H,W).

            data_range:
                Per-sample range with shape (B,).

        Returns:
            Tensor with shape
            (B,1,H-win_size+1,W-win_size+1).
        """
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(
                "SSIM expects X and Y with shape (B,H,W), "
                f"got X={X.shape}, Y={Y.shape}"
            )

        if X.shape != Y.shape:
            raise ValueError(
                f"X and Y shapes differ: {X.shape} vs {Y.shape}"
            )

        if data_range.ndim == 0:
            data_range = data_range.unsqueeze(0)

        if data_range.shape[0] != X.shape[0]:
            raise ValueError(
                "data_range batch size does not match images: "
                f"{data_range.shape[0]} vs {X.shape[0]}"
            )

        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)

        data_range = data_range.to(
            device=X.device,
            dtype=X.dtype,
        )
        data_range = data_range[:, None, None, None]

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)

        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)

        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)

        numerator_luminance = 2 * ux * uy + C1
        numerator_contrast = 2 * vxy + C2

        denominator_luminance = ux**2 + uy**2 + C1
        denominator_contrast = vx + vy + C2

        return (
            numerator_luminance
            * numerator_contrast
            / (
                denominator_luminance
                * denominator_contrast
            )
        )

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
    ) -> torch.Tensor:
        ssim_map = self._ssim_map(
            X,
            Y,
            data_range,
        )

        return 1.0 - ssim_map.mean()


class ForegroundSSIMLoss(SSIMLoss):
    """Differentiable counterpart of leaderboard SSIM_full.

    The reconstruction and target are masked before SSIM calculation.
    The resulting SSIM map is averaged only over valid foreground pixels.
    """

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        foreground_mask: torch.Tensor,
    ) -> torch.Tensor:
        if foreground_mask.ndim == 2:
            foreground_mask = foreground_mask.unsqueeze(0)

        if foreground_mask.shape != X.shape:
            raise ValueError(
                "foreground_mask and image shapes differ: "
                f"mask={foreground_mask.shape}, image={X.shape}"
            )

        foreground_mask = foreground_mask.to(
            device=X.device,
            dtype=X.dtype,
        )

        # This matches metrics.ssim_full:
        # SSIM is calculated after masking both images.
        ssim_map = self._ssim_map(
            X * foreground_mask,
            Y * foreground_mask,
            data_range,
        )

        # _ssim_map uses valid convolution, so remove win_size // 2
        # pixels from every border of the foreground mask.
        pad = self.win_size // 2

        if pad > 0:
            valid_mask = foreground_mask[
                :,
                pad:foreground_mask.shape[-2] - pad,
                pad:foreground_mask.shape[-1] - pad,
            ]
        else:
            valid_mask = foreground_mask

        valid_mask = valid_mask.unsqueeze(1)

        denominator = valid_mask.sum(
            dim=(1, 2, 3)
        )
        valid_samples = denominator > 0

        if not torch.any(valid_samples):
            # Differentiable zero.
            return ssim_map.sum() * 0.0

        numerator = (
            (1.0 - ssim_map)
            * valid_mask
        ).sum(dim=(1, 2, 3))

        per_sample_loss = (
            numerator
            / denominator.clamp_min(1.0)
        )

        return per_sample_loss[valid_samples].mean()


class BBoxSSIMLoss(SSIMLoss):
    """Average SSIM loss over individual annotation boxes.

    Every valid box receives equal weight, matching leaderboard SSIM_bbox.

    The preferred input is:

        boxes_batch = [
            [box_0, box_1, ...],  # sample 0
            [box_0, ...],         # sample 1
        ]

    Each box is a dictionary containing:

        x, y, width, height, label

    A legacy union-mask tensor is also accepted so existing smoke tests remain
    usable while the training pipeline is migrated.
    """

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        boxes_batch,
    ) -> torch.Tensor:
        # Backward compatibility for the previous bbox-mask smoke test.
        if torch.is_tensor(boxes_batch):
            return self._forward_union_mask(
                X,
                Y,
                data_range,
                boxes_batch,
            )

        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(
                "BBoxSSIMLoss expects X and Y with shape (B,H,W), "
                f"got X={X.shape}, Y={Y.shape}"
            )

        if X.shape != Y.shape:
            raise ValueError(
                f"X and Y shapes differ: {X.shape} vs {Y.shape}"
            )

        if len(boxes_batch) != X.shape[0]:
            raise ValueError(
                "boxes_batch length does not match image batch size: "
                f"{len(boxes_batch)} vs {X.shape[0]}"
            )

        if data_range.ndim == 0:
            data_range = data_range.unsqueeze(0)

        if data_range.shape[0] != X.shape[0]:
            raise ValueError(
                "data_range batch size does not match images: "
                f"{data_range.shape[0]} vs {X.shape[0]}"
            )

        box_losses = []

        image_height = X.shape[-2]
        image_width = X.shape[-1]

        for batch_index, boxes in enumerate(boxes_batch):
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
                    image_width,
                    int(box["x"])
                    + int(box["width"]),
                )
                y1 = min(
                    image_height,
                    int(box["y"])
                    + int(box["height"]),
                )

                box_width = x1 - x0
                box_height = y1 - y0

                # This matches metrics.ssim_bbox. A crop smaller than the
                # SSIM window cannot be evaluated.
                if (
                    box_width < self.win_size
                    or box_height < self.win_size
                ):
                    continue

                reconstruction_crop = X[
                    batch_index:batch_index + 1,
                    y0:y1,
                    x0:x1,
                ]

                target_crop = Y[
                    batch_index:batch_index + 1,
                    y0:y1,
                    x0:x1,
                ]

                crop_ssim_map = self._ssim_map(
                    reconstruction_crop,
                    target_crop,
                    data_range[
                        batch_index:batch_index + 1
                    ],
                )

                # Each box contributes exactly one scalar loss.
                box_losses.append(
                    1.0 - crop_ssim_map.mean()
                )

        if not box_losses:
            # Keep the zero connected to X so backward() remains valid.
            return X.sum() * 0.0

        # Equal weight for every valid annotation box.
        return torch.stack(box_losses).mean()

    def _forward_union_mask(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        bbox_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Previous union-mask behavior retained for compatibility tests."""
        if bbox_mask.ndim == 2:
            bbox_mask = bbox_mask.unsqueeze(0)

        if bbox_mask.ndim != 3:
            raise ValueError(
                "bbox_mask must have shape (B,H,W), "
                f"got {bbox_mask.shape}"
            )

        if bbox_mask.shape != X.shape:
            raise ValueError(
                "bbox_mask and image shapes differ: "
                f"mask={bbox_mask.shape}, image={X.shape}"
            )

        ssim_map = self._ssim_map(
            X,
            Y,
            data_range,
        )

        bbox_mask = bbox_mask.to(
            device=X.device,
            dtype=X.dtype,
        ).unsqueeze(1)

        # A window is valid only when every pixel in that window lies inside
        # the union bbox mask.
        window_coverage = F.conv2d(
            bbox_mask,
            self.w,
        )

        valid_windows = (
            window_coverage >= 1.0 - 1e-6
        ).to(dtype=X.dtype)

        denominator = valid_windows.sum(
            dim=(1, 2, 3)
        )
        valid_samples = denominator > 0

        if not torch.any(valid_samples):
            return ssim_map.sum() * 0.0

        numerator = (
            (1.0 - ssim_map)
            * valid_windows
        ).sum(dim=(1, 2, 3))

        per_sample_loss = (
            numerator
            / denominator.clamp_min(1.0)
        )

        return per_sample_loss[valid_samples].mean()