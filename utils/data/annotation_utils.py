from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def center_crop_or_pad_2d(
    data: torch.Tensor,
    output_shape: Sequence[int],
) -> torch.Tensor:
    """Center-crop or zero-pad the last two dimensions.

    Args:
        data:
            Tensor whose last two dimensions are height and width.
            Supported examples:

                (H, W)
                (N, H, W)
                (B, N, H, W)

        output_shape:
            Desired ``(height, width)``.

    Returns:
        Tensor with the same leading dimensions and the requested
        spatial dimensions.
    """
    if data.ndim < 2:
        raise ValueError(
            f"Expected at least 2 dimensions, got {data.shape}"
        )

    output_height = int(output_shape[0])
    output_width = int(output_shape[1])

    if output_height <= 0 or output_width <= 0:
        raise ValueError(
            f"Invalid output shape: {output_shape}"
        )

    height, width = data.shape[-2:]

    # Zero-pad dimensions that are smaller than the requested size.
    pad_height = max(output_height - height, 0)
    pad_width = max(output_width - width, 0)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    if pad_height > 0 or pad_width > 0:
        data = F.pad(
            data,
            (
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
            ),
            mode="constant",
            value=0,
        )

    # Center-crop dimensions that are larger than the requested size.
    height, width = data.shape[-2:]

    start_y = (height - output_height) // 2
    start_x = (width - output_width) // 2

    return data[
        ...,
        start_y:start_y + output_height,
        start_x:start_x + output_width,
    ]


def boxes_to_masks(
    boxes: Sequence[Dict],
    image_shape: Sequence[int] = (384, 384),
) -> Tuple[torch.Tensor, List[str]]:
    """Convert annotation boxes into separate binary masks.

    Each box gets its own mask channel. Keeping boxes separate allows us
    to recover one transformed bounding box for each original lesion.

    Args:
        boxes:
            Sequence of dictionaries containing ``x``, ``y``, ``width``,
            ``height`` and optionally ``label``.

        image_shape:
            Coordinate space in which the boxes are defined. The 2026
            challenge annotations use a 384x384 image space.

    Returns:
        masks:
            Float tensor with shape ``(num_boxes, H, W)``.

        labels:
            Label corresponding to each mask channel.
    """
    height = int(image_shape[0])
    width = int(image_shape[1])

    masks = []
    labels = []

    for box in boxes:
        x0 = max(0, int(box["x"]))
        y0 = max(0, int(box["y"]))

        x1 = min(
            width,
            int(box["x"]) + int(box["width"]),
        )
        y1 = min(
            height,
            int(box["y"]) + int(box["height"]),
        )

        if x1 <= x0 or y1 <= y0:
            continue

        mask = torch.zeros(
            (height, width),
            dtype=torch.float32,
        )

        mask[y0:y1, x0:x1] = 1.0

        masks.append(mask)
        labels.append(str(box.get("label", "")))

    if not masks:
        return (
            torch.zeros(
                (0, height, width),
                dtype=torch.float32,
            ),
            [],
        )

    return torch.stack(masks, dim=0), labels


def masks_to_boxes(
    masks: torch.Tensor,
    labels: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    min_size: int = 1,
) -> List[Dict]:
    """Recover axis-aligned boxes from transformed binary masks.

    Rotation and shear can turn a rectangular mask into a polygon. This
    function returns the smallest axis-aligned rectangle containing that
    transformed region.

    Args:
        masks:
            Tensor with shape ``(N, H, W)`` or ``(H, W)``.

        labels:
            Optional label for each mask.

        threshold:
            Values above this threshold are considered part of the box.

        min_size:
            Discard boxes whose transformed width or height is smaller
            than this value.
    """
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks with shape (N,H,W), got {masks.shape}"
        )

    if labels is None:
        labels = [""] * masks.shape[0]

    if len(labels) != masks.shape[0]:
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match "
            f"number of masks ({masks.shape[0]})"
        )

    boxes = []

    for mask, label in zip(masks, labels):
        indices = torch.nonzero(
            mask > threshold,
            as_tuple=False,
        )

        if indices.numel() == 0:
            continue

        y0 = int(indices[:, 0].min().item())
        y1 = int(indices[:, 0].max().item()) + 1

        x0 = int(indices[:, 1].min().item())
        x1 = int(indices[:, 1].max().item()) + 1

        box_width = x1 - x0
        box_height = y1 - y0

        if box_width < min_size or box_height < min_size:
            continue

        boxes.append(
            {
                "x": x0,
                "y": y0,
                "width": box_width,
                "height": box_height,
                "label": str(label),
            }
        )

    return boxes


def target_masks_to_image_space(
    masks: torch.Tensor,
    image_shape: Sequence[int],
) -> torch.Tensor:
    """Map 384x384 annotation masks into the full coil-image space.

    MRAugment operates on the full image produced by IFFT, whereas the
    annotation coordinates refer to the center-cropped/padded 384x384
    target. This function reverses that center crop/pad operation.
    """
    return center_crop_or_pad_2d(
        masks,
        image_shape,
    )


def image_masks_to_target_space(
    masks: torch.Tensor,
    target_shape: Sequence[int] = (384, 384),
) -> torch.Tensor:
    """Map transformed full-image masks back to target coordinates."""
    return center_crop_or_pad_2d(
        masks,
        target_shape,
    )


def boxes_to_union_mask(
    boxes: Sequence[Dict],
    image_shape: Sequence[int] = (384, 384),
) -> torch.Tensor:
    """Return one binary mask containing every annotation box.

    This will later be useful for a mask-weighted bbox loss.
    """
    masks, _ = boxes_to_masks(
        boxes,
        image_shape=image_shape,
    )

    if masks.shape[0] == 0:
        return torch.zeros(
            tuple(image_shape),
            dtype=torch.float32,
        )

    return masks.amax(dim=0)