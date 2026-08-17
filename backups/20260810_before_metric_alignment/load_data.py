import json

import h5py
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from utils.data.transforms import DataTransform


def get_slice_boxes(attrs, slice_idx):
    """Return fastMRI+ annotation boxes for one slice.

    The image H5 stores annotations as a JSON string:

        {
            "<slice index>": [
                {
                    "x": ...,
                    "y": ...,
                    "width": ...,
                    "height": ...,
                    "label": ...
                }
            ]
        }

    Box coordinates are defined in the 384x384 target-image space.
    """
    raw = attrs.get("annotations", "{}")

    # h5py may return the JSON attribute as bytes or a NumPy scalar.
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        raw = raw.item()

    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")

    if isinstance(raw, str):
        try:
            annotations = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid annotations JSON at slice {slice_idx}"
            ) from exc
    elif isinstance(raw, dict):
        annotations = raw
    else:
        raise TypeError(
            f"Unsupported annotations type: {type(raw)}"
        )

    boxes = annotations.get(str(int(slice_idx)), [])
    normalized_boxes = []

    for box in boxes:
        x = int(box["x"])
        y = int(box["y"])
        width = int(box["width"])
        height = int(box["height"])

        # Ignore malformed empty boxes.
        if width <= 0 or height <= 0:
            continue

        normalized_boxes.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "label": box.get("label", ""),
            }
        )

    return normalized_boxes


class SliceData(Dataset):
    def __init__(
        self,
        root,
        transform,
        input_key,
        target_key,
        forward=False,
    ):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward

        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = sorted(
                Path(root / "image").glob("*.h5")
            )

            for fname in image_files:
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind)
                    for slice_ind in range(num_slices)
                ]

        kspace_files = sorted(
            Path(root / "kspace").glob("*.h5")
        )

        for fname in kspace_files:
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind)
                for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf:
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf:
                num_slices = hf[self.target_key].shape[0]
            else:
                raise KeyError(
                    f"{fname} contains neither "
                    f"{self.input_key!r} nor {self.target_key!r}"
                )

        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, image_slice = self.image_examples[i]

        kspace_fname, dataslice = self.kspace_examples[i]

        if not self.forward:
            if image_fname.name != kspace_fname.name:
                raise ValueError(
                    f"Image file {image_fname.name} does not match "
                    f"kspace file {kspace_fname.name}"
                )

            if image_slice != dataslice:
                raise ValueError(
                    f"Image slice {image_slice} does not match "
                    f"kspace slice {dataslice} in {kspace_fname.name}"
                )

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask = np.asarray(hf["mask"])

        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)

            # Store only the current slice's boxes in a normalized form.
            # DataTransform does not use this value yet, so the current
            # training return format remains unchanged.
            attrs["slice_boxes"] = get_slice_boxes(
                attrs,
                dataslice,
            )

        return self.transform(
            mask,
            input,
            target,
            attrs,
            kspace_fname.name,
            dataslice,
        )


def create_data_loaders(
    data_path,
    args,
    shuffle=False,
    isforward=False,
    mask_augmentor=None,
    augmentor=None,
):
    if not isforward:
        max_key = args.max_key
        target_key = args.target_key
    else:
        max_key = -1
        target_key = -1

        # Never augment validation/leaderboard inference data.
        mask_augmentor = None
        augmentor = None

    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(
            isforward,
            max_key,
            mask_augmentor=mask_augmentor,
            augmentor=augmentor,
        ),
        input_key=args.input_key,
        target_key=target_key,
        forward=isforward,
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )

    return data_loader