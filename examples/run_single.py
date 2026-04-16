"""
run_single.py -- Run Grad-CAM on one X-ray image and save the visualisation.

This is the simplest entry-point to the xray-gradcam pipeline.  It loads a
pre-trained DenseNet121 checkpoint, preprocesses a single image, computes a
Grad-CAM (or Grad-CAM++) heatmap, and writes a three-panel PNG showing the
original X-ray, the heatmap, and the blended overlay.

Typical usage
-------------
# Explain the model's top prediction:
python examples/run_single.py \
    --image      data/test_images/patient_01.jpg \
    --checkpoint checkpoints/best_model.pt

# Explain class 1 (Pneumonia) even if the model predicted Normal:
python examples/run_single.py \
    --image      data/test_images/patient_01.jpg \
    --checkpoint checkpoints/best_model.pt \
    --class-idx  1

# Use Grad-CAM++ for sharper localisation on a GPU machine:
python examples/run_single.py \
    --image      data/test_images/patient_01.jpg \
    --checkpoint checkpoints/best_model.pt \
    --method     gradcam++ \
    --device     cuda
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so that "src.*" imports resolve
# regardless of which directory the user launches the script from.
# ---------------------------------------------------------------------------
# __file__ is  .../xray-gradcam/examples/run_single.py
# .parent      -> .../xray-gradcam/examples/
# .parent.parent -> .../xray-gradcam/           <-- project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradcam import GradCAM, GradCAMPlusPlus
from src.model import get_target_layer, load_model, predict
from src.visualize import overlay_heatmap, preprocess_xray, save_visualization


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    """
    Define and return the command-line argument parser.

    Keeping the parser in its own function makes it easy to import and call
    from tests or other scripts without triggering the full main() logic.
    """
    parser = argparse.ArgumentParser(
        prog="run_single.py",
        description=(
            "Visualise which regions of an X-ray drove a DenseNet121 "
            "prediction using Grad-CAM or Grad-CAM++."
        ),
        # RawDescriptionHelpFormatter preserves the newlines in the epilog
        # below, so the 'Example usage' block prints neatly.
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/run_single.py --image data/test_images/sample.jpg\n"
            "  python examples/run_single.py --image img.jpg --method gradcam++ --device cuda\n"
        ),
    )

    # ------------------------------------------------------------------
    # Required argument
    # ------------------------------------------------------------------

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        # metavar controls what is shown in the --help output instead of
        # the raw uppercase argument name (IMAGE).  A realistic example
        # makes the help text immediately understandable.
        metavar="PATH",
        help=(
            "Path to the input X-ray image.  Accepts JPEG or PNG.  "
            "Greyscale images are automatically converted to RGB.  "
            "Example: data/test_images/patient_01.jpg"
        ),
    )

    # ------------------------------------------------------------------
    # Optional arguments with sensible defaults
    # ------------------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        metavar="PATH",
        help=(
            "Path to the saved model weights (.pt or .pth file produced by "
            "torch.save).  The file may contain a bare state-dict or a "
            "Lightning-style {'state_dict': ...} wrapper.  "
            "Default: checkpoints/best_model.pt"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.png",
        metavar="PATH",
        help=(
            "Where to write the three-panel visualisation PNG.  Parent "
            "directories are created automatically if they do not exist.  "
            "Default: outputs/result.png"
        ),
    )

    parser.add_argument(
        "--class-idx",
        type=int,
        default=None,
        metavar="INT",
        dest="class_idx",
        # dest='class_idx' maps the CLI flag --class-idx to a Python
        # attribute named class_idx (the hyphen is not valid in an
        # attribute name, so argparse would normally use class_idx anyway
        # but making it explicit avoids confusion).
        help=(
            "Index of the class to explain (0 = Normal, 1 = Pneumonia for "
            "the default two-class model).  When omitted, the class with the "
            "highest predicted probability is used -- i.e. the model's actual "
            "prediction is explained.  Use this flag to force the heatmap to "
            "show evidence *for* a specific class even if it was not predicted."
        ),
    )

    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=["gradcam", "gradcam++"],
        # choices enforces that only these two strings are accepted and
        # automatically adds them to the --help output.  Any other value
        # causes argparse to print an error and exit before main() runs.
        help=(
            "Which saliency method to use.  "
            "'gradcam'   -- standard Grad-CAM (Selvaraju et al. 2017): fast, "
            "broad region highlighting.  "
            "'gradcam++' -- Grad-CAM++ (Chattopadhay et al. 2018): sharper "
            "localisation, better when multiple lesions are present.  "
            "Default: gradcam"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,   # None triggers auto-detection in main()
        choices=["cpu", "cuda"],
        help=(
            "Device to run inference on.  'cuda' uses the first available "
            "NVIDIA GPU and is much faster for large images or batches.  "
            "When omitted, CUDA is used automatically if available, otherwise "
            "CPU is used.  Force 'cpu' if you want reproducible timing on a "
            "CPU-only machine."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main(args):
    """
    Execute the single-image Grad-CAM pipeline.

    Args:
        args (argparse.Namespace): Parsed command-line arguments from
            build_parser().parse_args().
    """

    # ---- 1. Resolve device -----------------------------------------------
    # Auto-detect GPU if the user did not specify --device.  torch.cuda is
    # only imported here (not at module level) so the script stays importable
    # even in environments where PyTorch is not installed -- useful for linting
    # and documentation builds.
    import torch

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[device]  Auto-detected: {device}")
    else:
        device = args.device
        print(f"[device]  Using: {device}")

    # ---- 2. Load model ---------------------------------------------------
    print(f"[model]   Loading weights from: {args.checkpoint}")
    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=2,    # Normal (0) and Pneumonia (1)
        device=device,
    )
    print("[model]   DenseNet121 loaded and set to eval mode.")

    # ---- 3. Preprocess image ---------------------------------------------
    print(f"[image]   Preprocessing: {args.image}")
    tensor, original_np = preprocess_xray(args.image, image_size=224)
    # tensor      : (1, 3, 224, 224) float32 -- ready for model inference
    # original_np : (224, 224, 3)    uint8   -- kept for overlay display

    # ---- 4. Select and initialise GradCAM --------------------------------
    target_layer = get_target_layer(model)
    # get_target_layer() returns model.features.denseblock4.denselayer16.conv2
    # -- the last conv layer before global average-pooling, which produces the
    # richest spatial feature maps for DenseNet121.

    # The 'with' block guarantees that hooks registered on target_layer are
    # cleaned up even if an exception is raised inside the block.
    GradCAMClass = GradCAMPlusPlus if args.method == "gradcam++" else GradCAM
    print(f"[gradcam] Using method: {args.method}")

    with GradCAMClass(model, target_layer) as cam:

        # ---- 5. Generate heatmap -----------------------------------------
        cam_array, explained_idx = cam.generate(tensor, class_idx=args.class_idx)
        # cam_array    : float32 numpy array, shape (7, 7) for 224px input,
        #                values normalised to [0, 1].
        # explained_idx: the class index whose score was back-propagated --
        #                either args.class_idx or the argmax of the logits.

    # Hooks are now removed (context manager __exit__ called remove_hooks()).

    # ---- 6. Predict (label + confidence) ---------------------------------
    # predict() runs a clean forward pass with torch.no_grad(), returns a
    # dict with 'predicted_class', 'confidence', and 'probabilities'.
    result = predict(model, tensor, device=device)
    label      = result["predicted_class"]
    confidence = result["confidence"]
    class_names = ["Normal", "Pneumonia"]
    explained_label = class_names[explained_idx] if explained_idx < len(class_names) else str(explained_idx)

    # ---- 7. Overlay heatmap on original image ----------------------------
    overlaid, colored_heatmap = overlay_heatmap(original_np, cam_array, alpha=0.4)
    # overlaid        : the X-ray with the colour heatmap blended on top
    # colored_heatmap : the JET-colourised heatmap alone (used in panel 2)

    # ---- 8. Save three-panel visualisation -------------------------------
    save_visualization(
        original=original_np,
        overlaid=overlaid,
        heatmap=colored_heatmap,
        prediction_label=label,
        confidence=confidence,
        save_path=args.output,
    )

    # ---- 9. Print summary to stdout --------------------------------------
    print(
        f"\nPrediction: {label} ({confidence:.1%}) | "
        f"Explained class: {explained_label} (idx {explained_idx})"
    )
    print(f"Heatmap saved to: {Path(args.output).resolve()}")


# ---------------------------------------------------------------------------
# EXAMPLE USAGE (no real model needed - for portfolio demonstration)
# python examples/run_single.py --image data/test_images/sample.jpg --device cpu
#
# To explain class 1 (Pneumonia) regardless of the model's prediction:
# python examples/run_single.py --image data/test_images/sample.jpg --class-idx 1
#
# To use Grad-CAM++ with a custom checkpoint and output path:
# python examples/run_single.py \
#     --image      data/test_images/sample.jpg \
#     --checkpoint checkpoints/best_model.pt \
#     --output     outputs/patient_01_gradcampp.png \
#     --method     gradcam++
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    main(args)
