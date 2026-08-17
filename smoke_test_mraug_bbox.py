"""CPU-only alignment test for MRAugment and 2026 annotation boxes.

The test creates a synthetic single-coil MRI whose non-zero rectangle exactly
matches one annotation box. Each MRAugment geometry is forced independently.
After augmentation, the bounding rectangle measured from the transformed MRI
target must agree with the box produced from the transformed annotation mask.

No dataset, GPU, checkpoint, or output file is used.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "utils" / "model"

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastmri import fft2c
from utils.data.mraugment.data_augment import AugmentationPipeline


HEIGHT = 384
WIDTH = 384
ORIGINAL_BOX = {
    "x": 132,
    "y": 151,
    "width": 61,
    "height": 43,
    "label": "synthetic-lesion",
}


def make_hparams(active_transform):
    weights = {
        "translation": 0.0,
        "rotation": 0.0,
        "shearing": 0.0,
        "scaling": 0.0,
        "rot90": 0.0,
        "fliph": 0.0,
        "flipv": 0.0,
    }
    weights[active_transform] = 1.0

    return SimpleNamespace(
        aug_weight_translation=weights["translation"],
        aug_weight_rotation=weights["rotation"],
        aug_weight_shearing=weights["shearing"],
        aug_weight_scaling=weights["scaling"],
        aug_weight_rot90=weights["rot90"],
        aug_weight_fliph=weights["fliph"],
        aug_weight_flipv=weights["flipv"],
        aug_upsample=False,
        aug_upsample_factor=2,
        aug_upsample_order=1,
        aug_interpolation_order=1,
        aug_max_translation_x=0.08,
        aug_max_translation_y=0.08,
        aug_max_rotation=15.0,
        aug_max_shearing_x=12.0,
        aug_max_shearing_y=12.0,
        aug_max_scaling=0.10,
    )


def make_synthetic_kspace():
    """Create one complex coil with a unit-valued rectangular lesion."""
    image = torch.zeros(
        1,
        HEIGHT,
        WIDTH,
        2,
        dtype=torch.float32,
    )

    x0 = ORIGINAL_BOX["x"]
    y0 = ORIGINAL_BOX["y"]
    x1 = x0 + ORIGINAL_BOX["width"]
    y1 = y0 + ORIGINAL_BOX["height"]

    image[0, y0:y1, x0:x1, 0] = 1.0
    return fft2c(image)


def support_box(target):
    """Measure an axis-aligned box from the transformed synthetic target."""
    threshold = 0.5 * float(target.max().item())
    foreground = target >= threshold
    coordinates = torch.nonzero(
        foreground,
        as_tuple=False,
    )

    if coordinates.numel() == 0:
        raise RuntimeError(
            "The transformed synthetic MRI has no foreground."
        )

    y0 = int(coordinates[:, 0].min().item())
    y1 = int(coordinates[:, 0].max().item()) + 1
    x0 = int(coordinates[:, 1].min().item())
    x1 = int(coordinates[:, 1].max().item()) + 1

    return {
        "x": x0,
        "y": y0,
        "width": x1 - x0,
        "height": y1 - y0,
    }


def coordinate_error(image_box, annotation_box):
    return {
        key: abs(
            int(image_box[key])
            - int(annotation_box[key])
        )
        for key in ("x", "y", "width", "height")
    }


def run_one_transform(transform_name):
    pipeline = AugmentationPipeline(
        make_hparams(transform_name),
        seed=2026,
    )
    pipeline.set_augmentation_strength(1.0)

    (
        _,
        transformed_target,
        transformed_boxes,
    ) = pipeline.augment_from_kspace(
        make_synthetic_kspace(),
        target_size=(HEIGHT, WIDTH),
        boxes=[ORIGINAL_BOX],
    )

    if not torch.isfinite(transformed_target).all():
        raise RuntimeError(
            f"{transform_name}: target contains NaN or Inf."
        )

    if len(transformed_boxes) != 1:
        raise RuntimeError(
            f"{transform_name}: expected one transformed box, "
            f"received {len(transformed_boxes)}."
        )

    transformed_box = transformed_boxes[0]
    if transformed_box.get("label") != ORIGINAL_BOX["label"]:
        raise RuntimeError(
            f"{transform_name}: annotation label was not preserved."
        )

    measured_box = support_box(transformed_target)
    errors = coordinate_error(
        measured_box,
        transformed_box,
    )

    # The MRI uses bilinear interpolation while annotation masks use nearest
    # neighbour. A one-pixel boundary discrepancy is normal after affine
    # interpolation; two pixels is a conservative smoke-test tolerance.
    if max(errors.values()) > 2:
        raise RuntimeError(
            f"{transform_name}: MRI/annotation misalignment. "
            f"image_box={measured_box}, "
            f"annotation_box={transformed_box}, "
            f"errors={errors}"
        )

    print(
        f"{transform_name:11s} OK | "
        f"image={measured_box} | "
        f"annotation={transformed_box} | "
        f"max_error={max(errors.values())} px"
    )


def main():
    torch.manual_seed(2026)

    transforms = (
        "fliph",
        "flipv",
        "rot90",
        "translation",
        "rotation",
        "shearing",
        "scaling",
    )

    for transform_name in transforms:
        run_one_transform(transform_name)

    print("MRAugment bbox-alignment smoke test OK")
    print("All seven geometries stayed aligned within 2 pixels.")
    print("CPU only; no dataset, GPU, or checkpoint was used.")


if __name__ == "__main__":
    main()
