"""
visualize.py -- Pre-processing, heatmap overlay, and report generation for
               X-ray Grad-CAM visualisations.

Pipeline overview
-----------------
Raw image file
    |
    v preprocess_xray()
(tensor for inference, numpy array for display)
    |
    +-> model + gradcam.generate() --> cam_array   (H_feat x W_feat float32)
    |
    v overlay_heatmap()
(overlaid_image, coloured_heatmap)
    |
    v save_visualization()
PNG saved to disk

For batch runs, generate_batch_report() chains the whole pipeline and emits
a summary table to stdout.
"""

from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Switch to the non-interactive Agg backend so save_visualization() works in
# headless environments (servers, CI, Docker) where no display is attached.
#
# Guard: only call matplotlib.use() if no backend has been set yet.
# Calling it after another module has already initialised a backend raises a
# warning (or in older matplotlib versions, a hard error).  This pattern is
# the safest way to request Agg without breaking callers that import
# visualize.py into an interactive notebook or a GUI application.
if matplotlib.get_backend().lower() != "agg":
    try:
        matplotlib.use("Agg")
    except Exception:
        # If switching fails (e.g. a figure already exists), continue with
        # whatever backend is active -- save_visualization() will still work
        # as long as the active backend supports savefig().
        pass


# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
# DenseNet121 (and virtually every torchvision model) was pre-trained on
# ImageNet.  Even when fine-tuned on X-rays, the convolutional filters still
# expect input in the same normalised range, because that is the distribution
# the weights were initialised for.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 1. preprocess_xray
# ---------------------------------------------------------------------------

def preprocess_xray(image_path, image_size=224):
    """
    Load an X-ray image from disk and produce both an inference-ready tensor
    and a display-ready numpy array.

    Two outputs are needed because the model requires a normalised float
    tensor while the visualisation code needs pixel values in [0, 255].  We
    derive both from the same PIL image so they are always pixel-perfectly
    aligned.

    Preprocessing pipeline applied to the tensor:

    1. **Resize** to (image_size, image_size) -- DenseNet121 was trained on
       224x224; deviating changes the receptive field of every layer.
    2. **ToTensor** -- converts HxWxC uint8 in [0, 255] to CxHxW float32
       in [0.0, 1.0].
    3. **Normalize** -- subtracts ImageNet channel means and divides by
       standard deviations.  Without this, activations at the first layer are
       orders of magnitude larger than what the pre-trained weights expect,
       producing garbage Grad-CAM maps.

    Args:
        image_path (str | os.PathLike):
            Path to the input image.  JPEG, PNG, and DICOM-converted PNGs are
            all acceptable.  Greyscale images are converted to RGB (3-channel)
            so the model's first conv layer receives the expected input shape.
        image_size (int, optional):
            Side length (pixels) to which both spatial dimensions are resized.
            Must match the size used during model training.  Defaults to 224.

    Returns:
        tuple(torch.Tensor, numpy.ndarray):

        * **tensor** -- float32 tensor of shape (1, 3, image_size, image_size)
          suitable for direct model input.
        * **original_np** -- uint8 numpy array of shape (image_size,
          image_size, 3) in RGB order, suitable for matplotlib display and
          cv2 overlay.

    Example:
        >>> tensor, original_np = preprocess_xray("data/test_images/cxr.jpg")
        >>> tensor.shape
        torch.Size([1, 3, 224, 224])
        >>> original_np.shape
        (224, 224, 3)
    """
    # Open and force to RGB.
    # X-rays are often stored as greyscale (PIL mode "L") or with an alpha
    # channel (RGBA).  Converting to RGB guarantees 3 channels so the model's
    # first Conv2d (which expects in_channels=3) never raises a shape error.
    image = Image.open(image_path).convert("RGB")

    # Resize before snapshotting the numpy array so that original_np and the
    # tensor always have identical spatial dimensions.  overlay_heatmap() can
    # then assume both are the same size and skip a redundant resize.
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)

    # Snapshot the display copy *before* normalisation.
    # np.array() on a PIL Image gives uint8 values in [0, 255], shape HxWxC.
    # The explicit .copy() ensures a contiguous, writable numpy array rather
    # than a read-only view into PIL's internal buffer.
    original_np = np.array(image, dtype=np.uint8).copy()   # (H, W, 3)

    # Build and apply the inference pipeline.
    pipeline = transforms.Compose([
        # ToTensor: PIL (HxWxC, uint8) -> torch (CxHxW, float32, /255).
        # The axis permutation is handled internally -- no manual transpose
        # needed.
        transforms.ToTensor(),
        # Normalize: shifts channel distributions to match what ImageNet-
        # pretrained weights saw, improving gradient signal quality during
        # Grad-CAM backprop.
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    tensor = pipeline(image)      # (3, H, W)
    tensor = tensor.unsqueeze(0)  # (1, 3, H, W) -- batch dimension required by model

    return tensor, original_np


# ---------------------------------------------------------------------------
# 2. overlay_heatmap
# ---------------------------------------------------------------------------

def overlay_heatmap(original_image, cam_array, alpha=0.4):
    """
    Resize a Grad-CAM activation map, colourise it, and blend it over the
    original X-ray.

    The CAM from GradCAM.generate() has the spatial resolution of the target
    convolutional layer (typically 7x7 for a 224x224 DenseNet121 input).
    This function upsamples it to match the display image, colourises it with
    COLORMAP_JET, and alpha-blends the result over the original.

    Why COLORMAP_JET?
    -----------------
    JET maps low activation -> blue, high activation -> red, which is the
    conventional colour scheme in the Grad-CAM literature and gives
    radiologists an immediately familiar "hot-spot" view.

    Args:
        original_image (numpy.ndarray):
            uint8 RGB array of shape (H, W, 3).  Typically the original_np
            output of preprocess_xray().
        cam_array (numpy.ndarray):
            float32 array of shape (H_cam, W_cam) with values in [0, 1].
            The raw output of GradCAM.generate().
        alpha (float, optional):
            Blending weight for the heatmap layer.  0.0 = heatmap invisible;
            1.0 = original invisible.  Defaults to 0.4 -- enough to highlight
            regions without obscuring lung structure.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray):

        * **overlaid** -- uint8 RGB array (H, W, 3): the original X-ray with
          the colour heatmap blended on top.
        * **colored_heatmap** -- uint8 RGB array (H, W, 3): the colourised
          heatmap alone (no original image), used as a side-by-side panel.

    Example:
        >>> overlaid, heatmap = overlay_heatmap(original_np, cam_array)
        >>> overlaid.shape
        (224, 224, 3)
    """
    h, w = original_image.shape[:2]

    # Upsample the CAM from feature resolution (e.g. 7x7) to display
    # resolution.  cv2.resize takes (width, height) -- reversed vs numpy
    # shape convention.  INTER_LINEAR gives smooth bilinear interpolation;
    # INTER_NEAREST would produce a visible blocky grid artefact.
    cam_resized = cv2.resize(cam_array, (w, h), interpolation=cv2.INTER_LINEAR)

    # float32 [0, 1] -> uint8 [0, 255].
    # np.clip guards against rare floating-point values marginally outside
    # [0, 1] (e.g. 1.0000001) that would wrap around silently after astype.
    cam_uint8 = np.clip(cam_resized * 255.0, 0, 255).astype(np.uint8)

    # cv2.applyColorMap outputs BGR because OpenCV's native channel order is
    # BGR.  We convert to RGB immediately so every downstream operation
    # (matplotlib, PIL, addWeighted) works in a consistent colour space.
    colored_heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # addWeighted formula: dst = alpha*src1 + (1-alpha)*src2 + gamma
    # We want:             dst = alpha*heatmap + (1-alpha)*original
    # gamma=0 means no additive brightness offset.
    # Both inputs must be uint8 arrays of the same shape -- guaranteed above.
    overlaid = cv2.addWeighted(
        colored_heatmap, alpha,
        original_image,  1.0 - alpha,
        0,
    )

    return overlaid, colored_heatmap


# ---------------------------------------------------------------------------
# 3. save_visualization
# ---------------------------------------------------------------------------

def save_visualization(
    original,
    overlaid,
    heatmap,
    prediction_label,
    confidence,
    save_path="outputs/result.png",
):
    """
    Create and save a three-panel diagnostic figure.

    Layout (left to right)::

        [ Original X-ray ]  [ Grad-CAM Heatmap ]  [ Overlay + Prediction ]

    The prediction title in panel 3 is colour-coded:

    * Red  (#d63c3c) for Pneumonia -- flags a pathological finding.
    * Green (#2e7d32) for Normal   -- calm signal for a healthy study.

    Any class name other than "Normal" is treated as pathological (red), so
    the function generalises to multi-class models without a hard-coded list.

    Args:
        original (numpy.ndarray):
            uint8 RGB array (H, W, 3) -- the unmodified X-ray.
        overlaid (numpy.ndarray):
            uint8 RGB array (H, W, 3) -- X-ray with heatmap blended in.
        heatmap (numpy.ndarray):
            uint8 RGB array (H, W, 3) -- colourised Grad-CAM alone.
        prediction_label (str):
            Human-readable class name, e.g. "Normal" or "Pneumonia".
        confidence (float):
            Softmax probability of the predicted class, in [0, 1].
        save_path (str | os.PathLike, optional):
            Destination path for the PNG file.  Parent directories are created
            automatically.  Defaults to 'outputs/result.png'.

    Returns:
        None.  Prints the absolute path of the saved file to stdout.

    Example:
        >>> save_visualization(orig, overlay, hmap, "Pneumonia", 0.943,
        ...                    save_path="outputs/case_01.png")
        Saved visualisation -> /absolute/path/outputs/case_01.png
    """
    save_path = Path(save_path)
    # exist_ok=True avoids a race condition if another process creates the
    # directory between the check and the mkdir call.
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Prediction title ------------------------------------------------
    title_color  = "#2e7d32" if prediction_label == "Normal" else "#d63c3c"
    overlay_title = f"Prediction: {prediction_label} ({confidence:.1%})"

    # ---- Build figure ----------------------------------------------------
    # figsize=(15, 5): ~5 inches per panel at default DPI -- enough detail
    # without producing impractically large files.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1 -- original X-ray
    axes[0].imshow(original)
    axes[0].set_title("Original X-ray", fontsize=13, pad=8)
    axes[0].axis("off")

    # Panel 2 -- colourised heatmap with a labelled colorbar.
    # We show the RGB heatmap array directly and attach a ScalarMappable
    # colorbar that mirrors the JET mapping used in overlay_heatmap().
    # Without the ScalarMappable, matplotlib cannot auto-derive a colorbar
    # from a plain RGB array.
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13, pad=8)
    axes[1].axis("off")

    sm = plt.cm.ScalarMappable(
        cmap="jet",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])   # required by matplotlib; no actual data needed here
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Activation strength", fontsize=10)

    # Replace raw float ticks (0.0 / 0.5 / 1.0) with named levels so the
    # scale is immediately readable without prior familiarity with CAM values.
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Med", "High"])

    # Panel 3 -- overlay with colour-coded, bold prediction title
    axes[2].imshow(overlaid)
    axes[2].set_title(
        overlay_title, fontsize=13, pad=8,
        color=title_color, fontweight="bold",
    )
    axes[2].axis("off")

    # ---- Save and release ------------------------------------------------
    # dpi=150 -> ~2250x750 px output: sharp for reports, not bloated.
    # bbox_inches='tight' removes surrounding whitespace.
    # plt.close() releases figure memory -- critical inside batch loops where
    # dozens of figures would otherwise accumulate and exhaust RAM.
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualisation -> {save_path.resolve()}")


# ---------------------------------------------------------------------------
# 4. generate_batch_report
# ---------------------------------------------------------------------------

def generate_batch_report(
    image_paths,
    model,
    gradcam,
    output_dir="outputs/batch/",
):
    """
    Run the full Grad-CAM pipeline over a list of images and emit a summary.

    For each image in image_paths this function:

    1. preprocess_xray()     -- load and normalise.
    2. gradcam.generate()    -- forward + backward pass, produce CAM.
    3. predict() from model  -- softmax label and confidence score.
    4. overlay_heatmap()     -- colourise and blend.
    5. save_visualization()  -- write the three-panel PNG to disk.
    6. Append to results list.

    A formatted summary table is printed to stdout after all images are
    processed.

    Args:
        image_paths (list[str | os.PathLike]):
            Paths to the input images.  Mixed JPEG / PNG is fine.
        model (torch.nn.Module):
            Loaded, eval-mode DenseNet121.  Used by predict() to obtain
            softmax probabilities after gradcam.generate() has already run
            the forward pass.
        gradcam (GradCAM | GradCAMPlusPlus):
            An already-initialised Grad-CAM object with hooks registered.
            The caller is responsible for hook lifecycle -- use a ``with``
            block or call remove_hooks() after this function returns.
        output_dir (str | os.PathLike, optional):
            Directory where per-image PNGs are written.  Created
            automatically if absent.  Defaults to 'outputs/batch/'.

    Returns:
        list[dict]: One dict per image, with keys:

        * "image"        (str)   -- original image path.
        * "prediction"   (str)   -- predicted class name.
        * "confidence"   (float) -- softmax confidence in [0, 1].
        * "output_path"  (str)   -- absolute path to the saved PNG.

    Example:
        >>> paths = list(Path("data/test_images").glob("*.jpg"))
        >>> with GradCAM(model, get_target_layer(model)) as cam:
        ...     results = generate_batch_report(paths, model, cam)
        >>> results[0]["prediction"]
        'Pneumonia'
    """
    # Relative import works whether src/ is used as a plain directory
    # (sys.path hack) or installed as a proper package via pip install -e .
    # The old absolute 'from src.model import predict' broke when the installed
    # package name differed from the directory name.
    from .model import predict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # next(model.parameters()).device is the canonical device query -- works
    # regardless of whether the model is on CPU, a single GPU, or DataParallel.
    device = next(model.parameters()).device

    for image_path in image_paths:
        image_path = Path(image_path)

        # -- 1. Pre-process ------------------------------------------------
        tensor, original_np = preprocess_xray(str(image_path))

        # -- 2. Grad-CAM ---------------------------------------------------
        # generate() runs its own forward + backward pass internally.
        # class_idx=None tells it to explain the top predicted class.
        cam_array, _class_idx = gradcam.generate(tensor, class_idx=None)

        # -- 3. Predict (label + confidence) ------------------------------
        # predict() from model.py handles device placement, no_grad, and
        # softmax.  Passing the same tensor ensures the reported confidence
        # matches the class that was visualised.
        result_dict = predict(model, tensor, device=device)
        label      = result_dict["predicted_class"]
        confidence = result_dict["confidence"]

        # -- 4. Overlay ----------------------------------------------------
        overlaid, colored_heatmap = overlay_heatmap(original_np, cam_array)

        # -- 5. Save -------------------------------------------------------
        # Output name: {original_stem}_gradcam.png
        # e.g. "patient_042.jpg" -> "patient_042_gradcam.png"
        out_path = output_dir / f"{image_path.stem}_gradcam.png"

        save_visualization(
            original=original_np,
            overlaid=overlaid,
            heatmap=colored_heatmap,
            prediction_label=label,
            confidence=confidence,
            save_path=out_path,
        )

        # -- 6. Collect result ---------------------------------------------
        results.append({
            "image":       str(image_path),
            "prediction":  label,
            "confidence":  confidence,
            "output_path": str(out_path.resolve()),
        })

    # ---- Summary table ---------------------------------------------------
    # Column widths are computed from actual data so long filenames never
    # truncate the layout.  Header widths act as minimum column widths.
    col_file = max(
        len("Filename"),
        max((len(Path(r["image"]).name) for r in results), default=0),
    )
    col_pred = max(
        len("Prediction"),
        max((len(r["prediction"]) for r in results), default=0),
    )
    col_conf = len("Confidence")

    header    = (f"{'Filename':<{col_file}}  "
                 f"{'Prediction':<{col_pred}}  "
                 f"{'Confidence':>{col_conf}}")
    separator = "-" * len(header)

    print(f"\n{separator}")
    print(header)
    print(separator)
    for r in results:
        print(
            f"{Path(r['image']).name:<{col_file}}  "
            f"{r['prediction']:<{col_pred}}  "
            f"{r['confidence']:>{col_conf}.1%}"
        )
    print(separator)
    print(f"Total: {len(results)} image(s) processed.\n")

    return results
