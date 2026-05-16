"""
run_batch.py -- Run Grad-CAM on every image in a directory and produce a
                summary report.

This script is the batch counterpart to run_single.py.  It discovers all
JPEG and PNG files under --image-dir, runs the full Grad-CAM pipeline on
each one, writes individual three-panel PNGs to --output-dir, and prints a
summary table to stdout.

Typical usage
-------------
# Process a whole folder with default settings:
python examples/run_batch.py \
    --image-dir  data/test_images/ \
    --checkpoint checkpoints/best_model.pt

# Use Grad-CAM++ and force GPU:
python examples/run_batch.py \
    --image-dir  data/test_images/ \
    --checkpoint checkpoints/best_model.pt \
    --method     gradcam++ \
    --device     cuda \
    --output-dir outputs/batch_gradcampp/
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so 'src.*' imports resolve when the
# script is run from any working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradcam import GradCAM, GradCAMPlusPlus
from src.model import get_target_layer, load_model
from src.visualize import generate_batch_report


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    """Define and return the CLI argument parser for batch processing."""
    parser = argparse.ArgumentParser(
        prog="run_batch.py",
        description=(
            "Run Grad-CAM or Grad-CAM++ on every X-ray in a directory and "
            "save three-panel visualisation PNGs with a summary table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/run_batch.py --image-dir data/test_images/\n"
            "  python examples/run_batch.py --image-dir data/ --method gradcam++ --device cuda\n"
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        metavar="DIR",
        dest="image_dir",
        help=(
            "Directory containing input X-ray images.  All .jpg, .jpeg, and "
            ".png files found directly inside this directory will be processed."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        metavar="PATH",
        help=(
            "Path to the saved model weights (.pt or .pth).  "
            "Default: checkpoints/best_model.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/batch/",
        metavar="DIR",
        dest="output_dir",
        help=(
            "Directory where per-image PNG files are written.  Created "
            "automatically if it does not exist.  "
            "Default: outputs/batch/"
        ),
    )

    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=["gradcam", "gradcam++"],
        help=(
            "'gradcam' (default) for broad region highlighting; "
            "'gradcam++' for sharper, multi-lesion localisation."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help=(
            "Device for inference.  Auto-detects CUDA if omitted.  "
            "Force 'cpu' for reproducible CPU-only benchmarks."
        ),
    )

    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        metavar="EXT",
        help=(
            "File extensions to include (space-separated).  "
            "Default: .jpg .jpeg .png"
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main(args):
    """Run the batch Grad-CAM pipeline."""
    import torch

    # ---- 1. Resolve device -----------------------------------------------
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[device]  Auto-detected: {device}")
    else:
        device = args.device
        print(f"[device]  Using: {device}")

    # ---- 2. Discover images ----------------------------------------------
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"[error]   --image-dir '{image_dir}' is not a directory.")
        sys.exit(1)

    extensions = {ext.lower() for ext in args.extensions}
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not image_paths:
        print(
            f"[warning] No images with extensions {sorted(extensions)} "
            f"found in '{image_dir}'."
        )
        sys.exit(0)

    print(f"[images]  Found {len(image_paths)} image(s) in '{image_dir}'")

    # ---- 3. Load model ---------------------------------------------------
    print(f"[model]   Loading weights from: {args.checkpoint}")
    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=2,
        device=device,
    )
    print("[model]   DenseNet121 loaded and set to eval mode.")

    # ---- 4. Run batch pipeline -------------------------------------------
    target_layer = get_target_layer(model)
    GradCAMClass = GradCAMPlusPlus if args.method == "gradcam++" else GradCAM
    print(f"[gradcam] Using method: {args.method}")

    with GradCAMClass(model, target_layer) as cam:
        results = generate_batch_report(
            image_paths=image_paths,
            model=model,
            gradcam=cam,
            output_dir=args.output_dir,
        )

    print(f"[done]    Results written to: {Path(args.output_dir).resolve()}")
    return results


# ---------------------------------------------------------------------------
# EXAMPLE USAGE (no real model needed - for portfolio demonstration)
# python examples/run_batch.py --image-dir data/test_images/ --device cpu
#
# Process with Grad-CAM++ and save to a custom output folder:
# python examples/run_batch.py \
#     --image-dir  data/test_images/ \
#     --checkpoint checkpoints/best_model.pt \
#     --method     gradcam++ \
#     --output-dir outputs/batch_pp/
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    main(args)
