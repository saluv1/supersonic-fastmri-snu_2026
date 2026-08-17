import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "utils" / "model"
COMMON_DIR = PROJECT_ROOT / "utils" / "common"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(1, str(MODEL_DIR))

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(1, str(COMMON_DIR))


from utils.learning.train_part import train
from utils.common.utils import seed_fix
from utils.data.mraugment.data_augment import DataAugmentor


def str_to_bool(value):
    return str(value).lower() == "true"


def parse():
    parser = argparse.ArgumentParser(
        description=(
            "Train PromptMR+ on FastMRI challenge Images"
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    # Basic training options.
    parser.add_argument(
        "-g",
        "--GPU-NUM",
        type=int,
        default=0,
        help="GPU number to allocate",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "-e",
        "--num-epochs",
        type=int,
        default=1,
        help="Number of epochs",
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=float,
        default=2e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "-r",
        "--report-interval",
        type=int,
        default=100,
        help="Report interval",
    )
    parser.add_argument(
        "-n",
        "--net-name",
        type=Path,
        default="test_promptmr",
        help="Name of network",
    )
    parser.add_argument(
        "-t",
        "--data-path-train",
        type=Path,
        default="/root/Data/train/",
        help="Directory of train data",
    )
    parser.add_argument(
        "-v",
        "--data-path-val",
        type=Path,
        default="/root/Data/val/",
        help="Directory of validation data",
    )

    # Learning-rate scheduler.
    parser.add_argument(
        "--lr-scheduler",
        type=str_to_bool,
        default=True,
        help="Enable MultiStepLR learning-rate decay",
    )
    parser.add_argument(
        "--lr-milestones",
        nargs="+",
        type=int,
        default=[16, 27],
        help=(
            "Epochs at which the learning rate is "
            "multiplied by lr-gamma"
        ),
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.3,
        help=(
            "Learning-rate multiplier applied at "
            "each milestone"
        ),
    )

    # Dataset keys.
    parser.add_argument(
        "--input-key",
        type=str,
        default="kspace",
        help="Name of input key",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="image_label",
        help="Name of target key",
    )
    parser.add_argument(
        "--max-key",
        type=str,
        default="max",
        help="Name of max key in attributes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=430,
        help="Fix random seed",
    )

    # Metric-aligned loss:
    #
    # foreground SSIM loss
    # + bbox_loss_weight * per-box SSIM loss
    parser.add_argument(
        "--bbox-loss-weight",
        type=float,
        default=0.3,
        help=(
            "Weight of per-box SSIM loss. "
            "Use 0 to disable bbox supervision."
        ),
    )

    # PromptMR+ hyperparameters.
    parser.add_argument(
        "--num-cascades",
        type=int,
        default=2,
        help=(
            "Number of cascades | "
            "12 in original PromptMR+"
        ),
    )
    parser.add_argument(
        "--n-feat0",
        type=int,
        default=8,
        help="feat0 | 48 in original",
    )
    parser.add_argument(
        "--feature-dim",
        nargs="+",
        type=int,
        default=[24, 32, 40],
        help=(
            "Feature dims | "
            "[72,96,120] in original"
        ),
    )
    parser.add_argument(
        "--prompt-dim",
        nargs="+",
        type=int,
        default=[8, 16, 24],
        help=(
            "Prompt dims | "
            "[24,48,72] in original"
        ),
    )
    parser.add_argument(
        "--sens-n-feat0",
        type=int,
        default=8,
        help="Sensitivity feat0 | 24 in original",
    )
    parser.add_argument(
        "--sens-feature-dim",
        nargs="+",
        type=int,
        default=[12, 16, 20],
        help="Sensitivity feature dimensions",
    )
    parser.add_argument(
        "--sens-prompt-dim",
        nargs="+",
        type=int,
        default=[4, 8, 12],
        help="Sensitivity prompt dimensions",
    )
    parser.add_argument(
        "--len-prompt",
        nargs="+",
        type=int,
        default=[3, 3, 3],
        help="Prompt vector count per level",
    )
    parser.add_argument(
        "--prompt-size",
        nargs="+",
        type=int,
        default=[16, 8, 4],
        help=(
            "Prompt spatial size | "
            "[64,32,16] in original"
        ),
    )
    parser.add_argument(
        "--n-enc-cab",
        nargs="+",
        type=int,
        default=[2, 3, 3],
        help="CABs in encoder blocks",
    )
    parser.add_argument(
        "--n-dec-cab",
        nargs="+",
        type=int,
        default=[2, 2, 3],
        help="CABs in decoder blocks",
    )
    parser.add_argument(
        "--n-skip-cab",
        nargs="+",
        type=int,
        default=[1, 1, 1],
        help="CABs in skip connections",
    )
    parser.add_argument(
        "--n-bottleneck-cab",
        type=int,
        default=3,
        help="CABs in bottleneck",
    )
    parser.add_argument(
        "--no-use-ca",
        action="store_true",
        help="Disable channel attention in CABs",
    )
    parser.add_argument(
        "--adaptive-input",
        type=str_to_bool,
        default=True,
        help="Residual adaptive input",
    )
    parser.add_argument(
        "--n-buffer",
        type=int,
        default=4,
        help="Feature buffers in PromptMRBlock",
    )
    parser.add_argument(
        "--n-history",
        type=int,
        default=3,
        help="History length | 11 in original",
    )
    parser.add_argument(
        "--use-sens-adj",
        type=str_to_bool,
        default=True,
        help=(
            "Adjacent slices for sensitivity map"
        ),
    )
    parser.add_argument(
        "-c",
        "--use-checkpoint",
        type=str_to_bool,
        default=True,
        help=(
            "Gradient checkpointing to reduce VRAM"
        ),
    )
    parser.add_argument(
        "--compute-sens-per-coil",
        type=str_to_bool,
        default=True,
        help=(
            "Compute sensitivity maps one coil at a time. "
            "True uses less VRAM."
        ),
    )

    # K-space sampling-mask augmentation.
    parser.add_argument(
        "--mask-aug",
        type=str_to_bool,
        default=False,
        help=(
            "Enable k-space sampling-mask augmentation"
        ),
    )
    parser.add_argument(
        "--mask-aug-weight",
        type=float,
        default=1.0,
        help=(
            "Maximum mask-augmentation probability"
        ),
    )
    parser.add_argument(
        "--mask-aug-start",
        type=int,
        default=0,
        help=(
            "First epoch with non-zero mask "
            "augmentation probability"
        ),
    )
    parser.add_argument(
        "--mask-aug-schedule",
        type=str,
        default="exp",
        choices=[
            "exp",
            "const",
        ],
    )
    parser.add_argument(
        "--mask-aug-plateau-epoch",
        type=int,
        default=10,
        help=(
            "Epoch at which mask augmentation "
            "reaches maximum probability"
        ),
    )
    parser.add_argument(
        "--mask-aug-accelerations",
        nargs="+",
        type=int,
        default=[4, 8],
        help=(
            "Acceleration factors used when "
            "regenerating masks"
        ),
    )
    parser.add_argument(
        "--mask-aug-random-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of augmented masks using "
            "random scattered lines. "
            "0 means equispaced masks only."
        ),
    )
    parser.add_argument(
        "--mask-aug-random-offset",
        type=str_to_bool,
        default=True,
        help=(
            "Randomize the phase of equispaced masks"
        ),
    )

    # Image-domain MRAugment options.
    parser = (
        DataAugmentor
        .add_augmentation_specific_args(
            parser
        )
    )

    parser.add_argument(
        "--annealing-epoch",
        type=int,
        default=1,
        help=(
            "Trailing epochs held at maximum "
            "MRAugment strength"
        ),
    )

    args = parser.parse_args()

    # Argument validation.
    if args.num_epochs <= 0:
        parser.error(
            "--num-epochs must be greater than 0"
        )

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be greater than 0"
        )

    if args.lr <= 0:
        parser.error(
            "--lr must be greater than 0"
        )

    if not 0 < args.lr_gamma <= 1:
        parser.error(
            "--lr-gamma must satisfy 0 < gamma <= 1"
        )

    if any(
        milestone <= 0
        for milestone in args.lr_milestones
    ):
        parser.error(
            "--lr-milestones must contain "
            "positive epoch numbers"
        )

    # Remove duplicates and enforce increasing order.
    args.lr_milestones = sorted(
        set(args.lr_milestones)
    )

    if args.bbox_loss_weight < 0:
        parser.error(
            "--bbox-loss-weight must be "
            "greater than or equal to 0"
        )

    if not 0 <= args.mask_aug_weight <= 1:
        parser.error(
            "--mask-aug-weight must satisfy "
            "0 <= weight <= 1"
        )

    if not 0 <= args.mask_aug_random_ratio <= 1:
        parser.error(
            "--mask-aug-random-ratio must satisfy "
            "0 <= ratio <= 1"
        )

    if (
        args.mask_aug_plateau_epoch
        < args.mask_aug_start
    ):
        parser.error(
            "--mask-aug-plateau-epoch must be "
            "greater than or equal to "
            "--mask-aug-start"
        )

    if args.annealing_epoch < 0:
        parser.error(
            "--annealing-epoch must be "
            "greater than or equal to 0"
        )

    if (
        getattr(
            args,
            "aug_weight_rot90",
            0.0,
        )
        > 0.0
    ):
        print(
            "[MRAugment] rot90 is restricted to "
            "180 degrees because H and W differ."
        )

    return args


if __name__ == "__main__":
    args = parse()

    if args.seed is not None:
        seed_fix(
            args.seed
        )

    result_root = (
        Path("../result")
        / args.net_name
    )

    args.exp_dir = (
        result_root
        / "checkpoints"
    )
    args.val_dir = (
        result_root
        / "reconstructions_val"
    )
    args.main_dir = (
        result_root
        / Path(__file__).name
    )
    args.val_loss_dir = result_root

    args.exp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.val_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train(args)