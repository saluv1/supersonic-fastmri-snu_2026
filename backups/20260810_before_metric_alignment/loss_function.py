"""
SSIM losses used for PromptMR training.

``SSIMLoss`` preserves the original full-image training loss.
``BBoxSSIMLoss`` averages SSIM error only over windows fully contained in
the fastMRI+ annotation boxes.
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
                Reconstruction with shape ``(B,H,W)``.
            Y:
                Target with shape ``(B,H,W)``.
            data_range:
                Per-sample range with shape ``(B,)``.

        Returns:
            Tensor with shape
            ``(B,1,H-win_size+1,W-win_size+1)``.
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
        return 1 - ssim_map.mean()


class BBoxSSIMLoss(SSIMLoss):
    """SSIM loss restricted to fastMRI+ annotation boxes.

    ``bbox_mask`` is a binary union of the current slice's transformed
    bounding boxes. Only SSIM windows fully contained in a box are used.
    This matches the valid-convolution behavior of computing SSIM on a box
    crop and avoids windows that cross a box boundary.

    Slices without boxes are skipped. If an entire batch has no boxes, this
    returns a differentiable zero.
    """

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        bbox_mask: torch.Tensor,
    ) -> torch.Tensor:
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

        # Average pooling with the SSIM kernel equals 1 only when every pixel
        # in the SSIM window is inside an annotation box.
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
            # Keep the result connected to X so backward() remains valid.
            return ssim_map.sum() * 0.0

        error_map = 1 - ssim_map
        numerator = (
            error_map
            * valid_windows
        ).sum(dim=(1, 2, 3))

        per_sample_loss = numerator / denominator.clamp_min(1.0)

        return per_sample_loss[valid_samples].mean()
