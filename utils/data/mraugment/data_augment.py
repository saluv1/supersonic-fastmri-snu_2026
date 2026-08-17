"""
MRAugment applies channel-by-channel random data augmentation to MRI slices.

This 2026 version optionally applies the exact same spatial transformation to
fastMRI+ annotation boxes. Boxes are converted to binary masks, transformed
with nearest-neighbour interpolation, and converted back to axis-aligned boxes
in the 384x384 target coordinate space.
"""

from math import exp

import numpy as np
import torch
import torchvision.transforms.functional as TF

from fastmri import fft2c, ifft2c, rss_complex

from utils.common.utils import center_crop
from utils.data.annotation_utils import (
    boxes_to_masks,
    image_masks_to_target_space,
    masks_to_boxes,
    target_masks_to_image_space,
)
from utils.data.mraugment.helpers import (
    complex_channel_first,
    complex_channel_last,
    complex_crop_if_needed,
    crop_if_needed,
)


class AugmentationPipeline:
    """Image-domain MRI augmentation with optional annotation propagation."""

    def __init__(self, hparams, seed=None):
        self.hparams = hparams
        self.weight_dict = {
            "translation": hparams.aug_weight_translation,
            "rotation": hparams.aug_weight_rotation,
            "scaling": hparams.aug_weight_scaling,
            "shearing": hparams.aug_weight_shearing,
            "rot90": hparams.aug_weight_rot90,
            "fliph": hparams.aug_weight_fliph,
            "flipv": hparams.aug_weight_flipv,
        }
        self.upsample_augment = hparams.aug_upsample
        self.upsample_factor = hparams.aug_upsample_factor
        self.upsample_order = hparams.aug_upsample_order
        self.transform_order = hparams.aug_interpolation_order
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState(seed)

    def seed(self, seed):
        """Reset the augmentation random-number generator."""
        self.rng = np.random.RandomState(seed)

    def random_apply(self, transform_name):
        probability = (
            self.weight_dict[transform_name]
            * self.augmentation_strength
        )
        return self.rng.uniform() < probability

    def set_augmentation_strength(self, probability):
        self.augmentation_strength = probability

    def augment_image(
        self,
        im,
        annotation_masks=None,
        max_output_size=None,
    ):
        """Apply identical geometry to a coil image and annotation masks.

        Args:
            im:
                Complex coil image with shape ``(C, H, W, 2)``.
            annotation_masks:
                Optional float tensor with shape ``(N, H, W)``. Each channel
                represents one annotation box in the full coil-image space.
            max_output_size:
                Optional final spatial crop size.

        Returns:
            Augmented image and optional augmented masks.
        """
        im = complex_channel_first(im)

        if annotation_masks is not None:
            if annotation_masks.ndim != 3:
                raise ValueError(
                    "annotation_masks must have shape (N,H,W), "
                    f"got {annotation_masks.shape}"
                )
            if annotation_masks.shape[-2:] != im.shape[-2:]:
                raise ValueError(
                    "Image and annotation-mask sizes differ: "
                    f"image={im.shape[-2:]}, "
                    f"masks={annotation_masks.shape[-2:]}"
                )
            annotation_masks = annotation_masks.to(
                device=im.device,
                dtype=torch.float32,
            )

        # Pixel-preserving transforms --------------------------------------

        if self.random_apply("fliph"):
            im = TF.hflip(im)
            if annotation_masks is not None:
                annotation_masks = TF.hflip(annotation_masks)

        if self.random_apply("flipv"):
            im = TF.vflip(im)
            if annotation_masks is not None:
                annotation_masks = TF.vflip(annotation_masks)

        # Only 180 degrees is allowed. Rotating 90/270 degrees swaps H/W and
        # breaks the anisotropic k-space/mask layout used by this challenge.
        if self.random_apply("rot90"):
            im = torch.rot90(im, 2, dims=(-2, -1))
            if annotation_masks is not None:
                annotation_masks = torch.rot90(
                    annotation_masks,
                    2,
                    dims=(-2, -1),
                )

        if self.random_apply("translation"):
            height, width = im.shape[-2:]

            translation_x = self.rng.uniform(
                -self.hparams.aug_max_translation_x,
                self.hparams.aug_max_translation_x,
            )
            translation_x = int(translation_x * height)

            translation_y = self.rng.uniform(
                -self.hparams.aug_max_translation_y,
                self.hparams.aug_max_translation_y,
            )
            translation_y = int(translation_y * width)

            pad, top, left = (
                self._get_translate_padding_and_crop(
                    im,
                    (translation_x, translation_y),
                )
            )

            # Reflection padding avoids a hard border in the MRI image.
            im = TF.pad(
                im,
                padding=pad,
                padding_mode="reflect",
            )
            im = TF.crop(
                im,
                top,
                left,
                height,
                width,
            )

            if annotation_masks is not None:
                # Regions outside the original annotation image are empty.
                annotation_masks = TF.pad(
                    annotation_masks,
                    padding=pad,
                    fill=0,
                    padding_mode="constant",
                )
                annotation_masks = TF.crop(
                    annotation_masks,
                    top,
                    left,
                    height,
                    width,
                )

        # Interpolating transforms -----------------------------------------

        interpolate = False

        if self.random_apply("rotation"):
            interpolate = True
            rotation = self.rng.uniform(
                -self.hparams.aug_max_rotation,
                self.hparams.aug_max_rotation,
            )
        else:
            rotation = 0.0

        if self.random_apply("shearing"):
            interpolate = True
            shear_x = self.rng.uniform(
                -self.hparams.aug_max_shearing_x,
                self.hparams.aug_max_shearing_x,
            )
            shear_y = self.rng.uniform(
                -self.hparams.aug_max_shearing_y,
                self.hparams.aug_max_shearing_y,
            )
        else:
            shear_x = 0.0
            shear_y = 0.0

        if self.random_apply("scaling"):
            interpolate = True
            scale = self.rng.uniform(
                1.0 - self.hparams.aug_max_scaling,
                1.0 + self.hparams.aug_max_scaling,
            )
        else:
            scale = 1.0

        upsample = interpolate and self.upsample_augment

        if upsample:
            original_shape = im.shape[-2:]
            upsampled_shape = [
                original_shape[0] * self.upsample_factor,
                original_shape[1] * self.upsample_factor,
            ]

            if self.upsample_order == 3:
                image_interpolation = (
                    TF.InterpolationMode.BICUBIC
                )
            else:
                image_interpolation = (
                    TF.InterpolationMode.BILINEAR
                )

            im = TF.resize(
                im,
                size=upsampled_shape,
                interpolation=image_interpolation,
            )

            if annotation_masks is not None:
                annotation_masks = TF.resize(
                    annotation_masks,
                    size=upsampled_shape,
                    interpolation=TF.InterpolationMode.NEAREST,
                )
        else:
            original_shape = None
            image_interpolation = TF.InterpolationMode.BILINEAR

        if interpolate:
            height, width = im.shape[-2:]
            pad = self._get_affine_padding_size(
                im,
                rotation,
                scale,
                (shear_x, shear_y),
            )

            im = TF.pad(
                im,
                padding=pad,
                padding_mode="reflect",
            )
            im = TF.affine(
                im,
                angle=rotation,
                scale=scale,
                shear=(shear_x, shear_y),
                translate=[0, 0],
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            im = TF.center_crop(im, (height, width))

            if annotation_masks is not None:
                annotation_masks = TF.pad(
                    annotation_masks,
                    padding=pad,
                    fill=0,
                    padding_mode="constant",
                )
                annotation_masks = TF.affine(
                    annotation_masks,
                    angle=rotation,
                    scale=scale,
                    shear=(shear_x, shear_y),
                    translate=[0, 0],
                    interpolation=TF.InterpolationMode.NEAREST,
                    fill=0,
                )
                annotation_masks = TF.center_crop(
                    annotation_masks,
                    (height, width),
                )

        if upsample:
            im = TF.resize(
                im,
                size=original_shape,
                interpolation=image_interpolation,
            )

            if annotation_masks is not None:
                annotation_masks = TF.resize(
                    annotation_masks,
                    size=original_shape,
                    interpolation=TF.InterpolationMode.NEAREST,
                )

        if max_output_size is not None:
            im = crop_if_needed(im, max_output_size)
            if annotation_masks is not None:
                annotation_masks = crop_if_needed(
                    annotation_masks,
                    max_output_size,
                )

        im = complex_channel_last(im)
        return im, annotation_masks

    def augment_from_kspace(
        self,
        kspace,
        target_size,
        boxes=None,
        max_train_size=None,
    ):
        """Augment k-space, its RSS target, and optional boxes together."""
        if kspace.is_cuda:
            kspace = kspace.cpu()

        im = ifft2c(kspace)
        annotation_masks = None
        annotation_labels = []

        if boxes is not None:
            target_masks, annotation_labels = boxes_to_masks(
                boxes,
                image_shape=target_size,
            )

            if target_masks.shape[0] > 0:
                annotation_masks = (
                    target_masks_to_image_space(
                        target_masks,
                        image_shape=im.shape[-3:-1],
                    )
                )

        im, annotation_masks = self.augment_image(
            im,
            annotation_masks=annotation_masks,
            max_output_size=max_train_size,
        )

        # Regenerate the target from the transformed coil images so the
        # augmented target stays pixel-aligned with the augmented k-space.
        target_tensor = rss_complex(im).unsqueeze(0)
        if target_tensor.is_cuda:
            target_tensor = target_tensor.cpu()

        target = center_crop(
            target_tensor,
            target_size[0],
            target_size[1],
        ).squeeze(0)

        kspace = fft2c(im)

        if kspace.is_cuda:
            kspace = kspace.cpu()
        if target.is_cuda:
            target = target.cpu()

        # Keep the original API until transforms.py starts passing boxes.
        if boxes is None:
            return kspace, target

        if annotation_masks is None:
            augmented_boxes = []
        else:
            target_masks = image_masks_to_target_space(
                annotation_masks,
                target_shape=target_size,
            )
            augmented_boxes = masks_to_boxes(
                target_masks,
                labels=annotation_labels,
                threshold=0.5,
                min_size=1,
            )

        return kspace, target, augmented_boxes

    @staticmethod
    def _get_affine_padding_size(im, angle, scale, shear):
        """Calculate padding required to avoid cropping an affine transform."""
        height, width = im.shape[-2:]
        corners = [
            [-height / 2, -width / 2, 1.0],
            [-height / 2, width / 2, 1.0],
            [height / 2, width / 2, 1.0],
            [height / 2, -width / 2, 1.0],
        ]

        matrix = torch.tensor(
            TF._get_inverse_affine_matrix(
                [0.0, 0.0],
                -angle,
                [0, 0],
                scale,
                [-value for value in shear],
            )
        ).reshape(2, 3)

        corners = torch.cat(
            [
                torch.tensor(corner).reshape(3, 1)
                for corner in corners
            ],
            dim=1,
        )
        transformed_corners = torch.matmul(matrix, corners)
        all_corners = torch.cat(
            [transformed_corners, corners[:2, :]],
            dim=1,
        )
        bounding_box = (
            all_corners.amax(dim=1)
            - all_corners.amin(dim=1)
        )

        pad_height = torch.clip(
            torch.floor((bounding_box[0] - height) / 2),
            min=0.0,
            max=height - 1,
        )
        pad_width = torch.clip(
            torch.floor((bounding_box[1] - width) / 2),
            min=0.0,
            max=width - 1,
        )

        # torchvision expects (horizontal padding, vertical padding).
        return int(pad_width.item()), int(pad_height.item())

    @staticmethod
    def _get_translate_padding_and_crop(im, translation):
        translation_x, translation_y = translation
        height, width = im.shape[-2:]
        pad = [0, 0, 0, 0]

        if translation_x >= 0:
            pad[3] = min(translation_x, height - 1)
            top = pad[3]
        else:
            pad[1] = min(-translation_x, height - 1)
            top = 0

        if translation_y >= 0:
            pad[0] = min(translation_y, width - 1)
            left = 0
        else:
            pad[2] = min(-translation_y, width - 1)
            left = pad[2]

        return pad, top, left


class DataAugmentor:
    """High-level augmentation pipeline and probability scheduler."""

    def __init__(self, hparams, current_epoch_fn, seed=None):
        self.current_epoch_fn = current_epoch_fn
        self.hparams = hparams
        self.aug_on = hparams.aug_on

        if self.aug_on:
            aug_seed = (
                seed + 1000
                if seed is not None
                else None
            )
            self.augmentation_pipeline = AugmentationPipeline(
                hparams,
                seed=aug_seed,
            )

        self.max_train_resolution = hparams.max_train_resolution

    def seed_pipeline(self, seed):
        if (
            self.aug_on
            and hasattr(self, "augmentation_pipeline")
        ):
            self.augmentation_pipeline.seed(seed)

    def __call__(
        self,
        kspace,
        target_size,
        boxes=None,
    ):
        """Generate augmented data with a backward-compatible return type.

        When ``boxes`` is ``None`` this returns ``(kspace, target)``.
        When boxes are supplied it returns ``(kspace, target, boxes)``.
        """
        if self.aug_on:
            probability = self.schedule_p()
            self.augmentation_pipeline.set_augmentation_strength(
                probability
            )
        else:
            probability = 0.0

        target = None
        augmented_boxes = boxes

        if self.aug_on and probability > 0.0:
            if boxes is None:
                kspace, target = (
                    self.augmentation_pipeline.augment_from_kspace(
                        kspace,
                        target_size=target_size,
                        max_train_size=self.max_train_resolution,
                    )
                )
            else:
                kspace, target, augmented_boxes = (
                    self.augmentation_pipeline.augment_from_kspace(
                        kspace,
                        target_size=target_size,
                        boxes=boxes,
                        max_train_size=self.max_train_resolution,
                    )
                )
        elif self.max_train_resolution is not None:
            too_tall = (
                kspace.shape[-3]
                > self.max_train_resolution[0]
            )
            too_wide = (
                kspace.shape[-2]
                > self.max_train_resolution[1]
            )

            if too_tall or too_wide:
                im = ifft2c(kspace)
                im = complex_crop_if_needed(
                    im,
                    self.max_train_resolution,
                )
                kspace = fft2c(im)

        if boxes is None:
            return kspace, target

        return kspace, target, augmented_boxes

    def schedule_p(self):
        delay = self.hparams.aug_delay
        plateau_epoch = self.hparams.max_epochs
        epoch = self.current_epoch_fn()
        maximum = self.hparams.aug_strength

        if epoch < delay:
            return 0.0

        if (
            self.hparams.aug_schedule == "constant"
            or epoch >= plateau_epoch
            or plateau_epoch <= delay
        ):
            return maximum

        if self.hparams.aug_schedule == "ramp":
            return (
                (epoch - delay)
                / (plateau_epoch - delay)
                * maximum
            )

        if self.hparams.aug_schedule == "exp":
            decay = (
                self.hparams.aug_exp_decay
                / (plateau_epoch - delay)
            )
            numerator = 1 - exp(
                -(epoch - delay) * decay
            )
            denominator = 1 - exp(
                -(plateau_epoch - delay) * decay
            )
            return maximum * numerator / denominator

        raise ValueError(
            "Unknown MRAugment schedule: "
            f"{self.hparams.aug_schedule!r}"
        )

    @staticmethod
    def add_augmentation_specific_args(parser):
        parser.add_argument(
            "--aug_on",
            default=False,
            action="store_true",
            help="Turn MRAugment on.",
        )

        parser.add_argument(
            "--aug_schedule",
            type=str,
            default="exp",
            choices=["constant", "ramp", "exp"],
            help="Augmentation-strength schedule.",
        )
        parser.add_argument(
            "--aug_delay",
            type=int,
            default=15,
            help="Initial epochs without MRAugment.",
        )
        parser.add_argument(
            "--aug_strength",
            type=float,
            default=0.5,
            help="Maximum augmentation strength.",
        )
        parser.add_argument(
            "--aug_exp_decay",
            type=float,
            default=5.0,
            help="Exponential-schedule decay coefficient.",
        )

        parser.add_argument(
            "--aug_interpolation_order",
            type=int,
            default=1,
            choices=[1, 3],
            help="Image interpolation order: 1=bilinear, 3=bicubic.",
        )
        parser.add_argument(
            "--aug_upsample",
            default=False,
            action="store_true",
            help="Upsample before affine augmentation.",
        )
        parser.add_argument(
            "--aug_upsample_factor",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--aug_upsample_order",
            type=int,
            default=1,
            choices=[1, 3],
        )

        parser.add_argument(
            "--aug_weight_translation",
            type=float,
            default=0.1,
        )
        parser.add_argument(
            "--aug_weight_rotation",
            type=float,
            default=0.1,
        )
        parser.add_argument(
            "--aug_weight_shearing",
            type=float,
            default=0.1,
        )
        parser.add_argument(
            "--aug_weight_scaling",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--aug_weight_rot90",
            type=float,
            default=0.0,
            help="Probability weight for a safe 180-degree rotation.",
        )
        parser.add_argument(
            "--aug_weight_fliph",
            type=float,
            default=0.4,
        )
        parser.add_argument(
            "--aug_weight_flipv",
            type=float,
            default=0.0,
        )

        parser.add_argument(
            "--aug_max_translation_x",
            type=float,
            default=0.05,
        )
        parser.add_argument(
            "--aug_max_translation_y",
            type=float,
            default=0.05,
        )
        parser.add_argument(
            "--aug_max_rotation",
            type=float,
            default=10.0,
        )
        parser.add_argument(
            "--aug_max_shearing_x",
            type=float,
            default=10.0,
        )
        parser.add_argument(
            "--aug_max_shearing_y",
            type=float,
            default=10.0,
        )
        parser.add_argument(
            "--aug_max_scaling",
            type=float,
            default=0.05,
        )

        parser.add_argument(
            "--max_train_resolution",
            nargs=2,
            default=None,
            type=int,
            metavar=("HEIGHT", "WIDTH"),
            help="Optional maximum training image size.",
        )

        return parser
