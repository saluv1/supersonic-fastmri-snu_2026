import numpy as np
import torch

from utils.data.annotation_utils import boxes_to_union_mask


def to_tensor(data):
    """Convert a NumPy array to a PyTorch tensor."""
    return torch.from_numpy(data)


class DataTransform:
    """Assemble one training, validation, or inference sample.

    Processing order:

    1. Read the original target and slice-level annotation boxes.
    2. Apply MRAugment to the fully sampled k-space, target, and boxes.
    3. Convert the resulting boxes to a fixed-size binary bbox mask.
    4. Optionally augment the k-space sampling mask.
    5. Apply the sampling mask and format the tensors for PromptMR.

    Training/validation return seven items:

        mask, kspace, target, maximum, fname, slice, bbox_mask

    Forward inference keeps the original six-item return format because
    annotation supervision is not used at inference time.
    """

    def __init__(
        self,
        isforward,
        max_key,
        mask_augmentor=None,
        augmentor=None,
    ):
        self.isforward = isforward
        self.max_key = max_key
        self.mask_augmentor = mask_augmentor
        self.augmentor = augmentor

    def __call__(
        self,
        mask,
        input,
        target,
        attrs,
        fname,
        slice,
    ):
        # H5 masks are expected to be one-dimensional along k-space width.
        mask = np.asarray(mask).reshape(-1)

        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
            boxes = attrs.get("slice_boxes", [])
        else:
            target = -1
            maximum = -1
            boxes = []

        # MRAugment works on fully sampled k-space. It returns a target and
        # boxes transformed with exactly the same spatial parameters.
        if (
            self.augmentor is not None
            and not self.isforward
            and self.augmentor.schedule_p() > 0.0
        ):
            kspace_full = to_tensor(input)
            kspace_full = torch.stack(
                (
                    kspace_full.real,
                    kspace_full.imag,
                ),
                dim=-1,
            )

            (
                augmented_kspace,
                augmented_target,
                augmented_boxes,
            ) = self.augmentor(
                kspace_full,
                tuple(target.shape),
                boxes=boxes,
            )

            if augmented_target is not None:
                input = (
                    augmented_kspace[..., 0]
                    + 1j * augmented_kspace[..., 1]
                ).numpy()
                target = augmented_target
                boxes = augmented_boxes

                # Keep the original volume-wide maximum. The official
                # evaluator uses the same H5 attribute as SSIM data_range.

        if not self.isforward:
            # A fixed HxW tensor collates safely even when slices contain
            # different numbers of boxes.
            bbox_mask = boxes_to_union_mask(
                boxes,
                image_shape=tuple(target.shape),
            )

        # Sampling-mask augmentation is performed after MRAugment so it is
        # applied to the newly generated full k-space.
        if self.mask_augmentor is not None:
            mask = self.mask_augmentor(mask)

        kspace = to_tensor(input * mask)
        kspace = torch.stack(
            (
                kspace.real,
                kspace.imag,
            ),
            dim=-1,
        )

        mask_tensor = torch.from_numpy(
            mask.reshape(
                1,
                1,
                kspace.shape[-2],
                1,
            ).astype(np.float32)
        ).byte()

        if self.isforward:
            return (
                mask_tensor,
                kspace,
                target,
                maximum,
                fname,
                slice,
            )

        return (
            mask_tensor,
            kspace,
            target,
            maximum,
            fname,
            slice,
            bbox_mask,
        )
