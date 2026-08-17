import h5py
import numpy as np
import torch

from collections import defaultdict

from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.data.transforms import to_tensor
from utils.learning.train_part import build_model


# recon_eval.py only calls load_model, prep_volume, and recon_slice.
INPUT_KIND = "kspace"


def _sens_mode(model):
    """Return the checkpoint-selected mode with an old-checkpoint fallback."""
    return bool(
        getattr(
            model,
            "compute_sens_per_coil",
            True,
        )
    )


def load_model(args, device):
    checkpoint = torch.load(
        args.exp_dir / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    train_args = checkpoint["args"]

    model = build_model(train_args).to(device=device)
    model.load_state_dict(checkpoint["model"])

    # Checkpoints made before this option was exposed remain compatible.
    model.compute_sens_per_coil = bool(
        getattr(
            train_args,
            "compute_sens_per_coil",
            True,
        )
    )
    model.eval()

    print(
        "[Sensitivity] inference compute_sens_per_coil="
        f"{model.compute_sens_per_coil}"
    )
    return model


def prep_volume(image_path, kspace_path, device):
    """Load one volume's k-space and mask onto the host."""
    with h5py.File(kspace_path, "r") as hf:
        kspace = hf["kspace"][:]
        mask = np.asarray(hf["mask"])

    return {
        "kspace": kspace,
        "mask": mask,
        "device": device,
        "num_slices": kspace.shape[0],
    }


def recon_slice(model, ctx, s):
    """Reconstruct a single slice with the model's selected sens mode."""
    device = ctx["device"]
    mask = np.asarray(ctx["mask"]).reshape(-1)

    kspace = to_tensor(
        ctx["kspace"][s] * mask
    )
    kspace = torch.stack(
        (
            kspace.real,
            kspace.imag,
        ),
        dim=-1,
    ).unsqueeze(0).to(device=device)

    mask_t = torch.from_numpy(
        mask.reshape(
            1,
            1,
            kspace.shape[-2],
            1,
        ).astype(np.float32)
    ).byte()
    mask_t = mask_t.unsqueeze(0).to(device=device)

    return model(
        kspace,
        mask_t,
        use_checkpoint=False,
        compute_sens_per_coil=_sens_mode(model),
    )[0]


def test(args, model, data_loader):
    """Legacy forward-loader path retained for reconstruct.py."""
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (
            mask,
            kspace,
            _,
            _,
            fnames,
            slices,
        ) in data_loader:
            kspace = kspace.to(
                device=next(model.parameters()).device,
                non_blocking=True,
            )
            mask = mask.to(
                device=next(model.parameters()).device,
                non_blocking=True,
            )
            output = model(
                kspace,
                mask,
                use_checkpoint=False,
                compute_sens_per_coil=_sens_mode(model),
            )

            for index in range(output.shape[0]):
                reconstructions[fnames[index]][
                    int(slices[index])
                ] = output[index].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [
                output
                for _, output in sorted(
                    reconstructions[fname].items()
                )
            ]
        )

    return reconstructions, None


def forward(args):
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

    model = load_model(args, device)

    forward_loader = create_data_loaders(
        data_path=args.data_path,
        args=args,
        isforward=True,
    )
    reconstructions, inputs = test(
        args,
        model,
        forward_loader,
    )
    save_reconstructions(
        reconstructions,
        args.forward_dir,
        inputs=inputs,
    )
