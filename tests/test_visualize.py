"""
test_visualize.py -- Unit tests for src/visualize.py.

Design principles
-----------------
* No real images on disk (except what we create ourselves in a tmp dir).
  PIL is used to write a synthetic solid-grey PNG to pytest's tmp_path
  fixture, so tests are self-contained and never depend on the presence of
  files in data/test_images/.

* No model or GradCAM objects required.
  preprocess_xray and overlay_heatmap are pure input-output functions;
  we only need numpy arrays and PIL images.

* save_visualization is tested with synthetic numpy arrays and pytest's
  tmp_path fixture -- no model needed, just PIL-writable numpy data.

* generate_batch_report is tested end-to-end using the TinyCNN fixture from
  conftest.py and a synthetic PNG written to tmp_path.  No real checkpoint
  or X-ray image is required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# sys.path guard
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualize import (
    generate_batch_report,
    overlay_heatmap,
    preprocess_xray,
    save_visualization,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_original():
    """
    A (224, 224, 3) uint8 numpy array of random pixel values.

    We use np.random.randint rather than np.zeros so the overlay maths
    exercises non-trivial blending (addWeighted on an all-black image would
    just return a scaled version of the heatmap, which could mask bugs).
    """
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def fake_cam():
    """
    A (7, 7) float32 array with values in [0, 1].

    7x7 matches the real spatial resolution of DenseNet121's last conv layer
    for a 224x224 input, so the resize step inside overlay_heatmap() is
    exercised rather than skipped.
    """
    rng = np.random.default_rng(seed=1)
    return rng.random((7, 7)).astype(np.float32)


@pytest.fixture
def grey_png(tmp_path):
    """
    Write a 256x256 solid-grey PNG to a temporary directory and return its
    path as a string.

    Why solid grey (128)?
    * Solid images are valid JPEG/PNG; they pass PIL's open without error.
    * A uniform colour is predictable: after ToTensor and Normalize the
      output values are deterministic, which lets test_preprocess_xray_normalized
      make a tight assertion (tensor.min() < 0) without depending on which
      specific pixels the RNG generates.
    * 256x256 != 224x224, so the Resize step is always exercised.

    Returns:
        str: Absolute path to the written PNG file.
    """
    img_path = tmp_path / "fake_xray.png"
    # Mode "RGB", size (256, 256), fill with grey (128, 128, 128)
    Image.new("RGB", (256, 256), color=(128, 128, 128)).save(img_path)
    return str(img_path)


# ---------------------------------------------------------------------------
# Test 1 -- overlay_heatmap output shape
# ---------------------------------------------------------------------------

class TestOverlayHeatmapOutputShape:
    """overlay_heatmap() must return two arrays whose spatial dims match the
    original image, regardless of the CAM's input resolution."""

    def test_overlaid_shape_matches_original(self, fake_original, fake_cam):
        """
        The overlaid image must be the same (H, W, C) as the original.
        If cv2.resize or addWeighted changes the shape, downstream
        matplotlib imshow() and save calls will raise ValueError.
        """
        overlaid, _ = overlay_heatmap(fake_original, fake_cam)

        assert overlaid.shape == fake_original.shape, (
            f"overlaid.shape={overlaid.shape} does not match "
            f"original.shape={fake_original.shape}"
        )

    def test_colored_heatmap_shape_matches_original(self, fake_original, fake_cam):
        """
        The standalone colourised heatmap must also be upsampled to the
        original's spatial dimensions -- it is displayed as a separate panel
        in save_visualization().
        """
        _, colored_heatmap = overlay_heatmap(fake_original, fake_cam)

        assert colored_heatmap.shape == fake_original.shape, (
            f"colored_heatmap.shape={colored_heatmap.shape} does not match "
            f"original.shape={fake_original.shape}"
        )

    def test_overlaid_is_3_channel(self, fake_original, fake_cam):
        """Both outputs must be 3-channel (RGB); single-channel would break imshow."""
        overlaid, colored_heatmap = overlay_heatmap(fake_original, fake_cam)

        assert overlaid.ndim == 3 and overlaid.shape[2] == 3, (
            f"overlaid should be (H, W, 3), got {overlaid.shape}"
        )
        assert colored_heatmap.ndim == 3 and colored_heatmap.shape[2] == 3, (
            f"colored_heatmap should be (H, W, 3), got {colored_heatmap.shape}"
        )

    def test_small_cam_is_upsampled_correctly(self, fake_original):
        """
        A 1x1 CAM (the smallest possible spatial map) must still be upsampled
        to the full original size without cv2.resize raising an error.
        """
        tiny_cam = np.array([[0.9]], dtype=np.float32)   # 1x1
        overlaid, colored_heatmap = overlay_heatmap(fake_original, tiny_cam)

        assert overlaid.shape == fake_original.shape
        assert colored_heatmap.shape == fake_original.shape


# ---------------------------------------------------------------------------
# Test 2 -- overlay_heatmap value range
# ---------------------------------------------------------------------------

class TestOverlayHeatmapValueRange:
    """All pixel values in both outputs must be valid uint8: in [0, 255]."""

    def test_overlaid_min_at_least_zero(self, fake_original, fake_cam):
        overlaid, _ = overlay_heatmap(fake_original, fake_cam)

        assert int(overlaid.min()) >= 0, (
            f"overlaid contains negative pixel values (min={overlaid.min()})"
        )

    def test_overlaid_max_at_most_255(self, fake_original, fake_cam):
        overlaid, _ = overlay_heatmap(fake_original, fake_cam)

        assert int(overlaid.max()) <= 255, (
            f"overlaid contains pixel values above 255 (max={overlaid.max()})"
        )

    def test_heatmap_min_at_least_zero(self, fake_original, fake_cam):
        _, colored_heatmap = overlay_heatmap(fake_original, fake_cam)

        assert int(colored_heatmap.min()) >= 0

    def test_heatmap_max_at_most_255(self, fake_original, fake_cam):
        _, colored_heatmap = overlay_heatmap(fake_original, fake_cam)

        assert int(colored_heatmap.max()) <= 255

    def test_overlaid_dtype_is_uint8(self, fake_original, fake_cam):
        """
        cv2.addWeighted returns uint8 when both inputs are uint8.  If our
        clip/astype pipeline accidentally produced float32, matplotlib's
        imshow would interpret values above 1.0 as clipped white, silently
        washing out the visualisation.
        """
        overlaid, colored_heatmap = overlay_heatmap(fake_original, fake_cam)

        assert overlaid.dtype == np.uint8, (
            f"overlaid.dtype={overlaid.dtype}, expected uint8"
        )
        assert colored_heatmap.dtype == np.uint8, (
            f"colored_heatmap.dtype={colored_heatmap.dtype}, expected uint8"
        )

    def test_all_black_original_still_valid(self, fake_cam):
        """
        Edge case: a fully black X-ray (e.g. failed scan) must not cause
        addWeighted to produce out-of-range values.
        """
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        overlaid, _ = overlay_heatmap(black, fake_cam)

        assert int(overlaid.min()) >= 0
        assert int(overlaid.max()) <= 255

    def test_cam_with_all_ones_still_valid(self, fake_original):
        """
        A saturated CAM (all 1.0 = maximum activation everywhere) must not
        wrap around or overflow after the clip + uint8 cast.
        """
        all_ones = np.ones((7, 7), dtype=np.float32)
        overlaid, heatmap = overlay_heatmap(fake_original, all_ones)

        assert int(overlaid.max()) <= 255
        assert int(heatmap.max()) <= 255


# ---------------------------------------------------------------------------
# Test 3 -- preprocess_xray tensor shape
# ---------------------------------------------------------------------------

class TestPreprocessXrayTensorShape:
    """preprocess_xray() must return correctly shaped outputs."""

    def test_tensor_batch_dim(self, grey_png):
        """
        The tensor must have a leading batch dimension of 1, because the
        model's first layer expects (N, C, H, W) not (C, H, W).
        """
        tensor, _ = preprocess_xray(grey_png, image_size=224)

        assert tensor.shape[0] == 1, (
            f"Expected batch dim=1, got tensor.shape={tensor.shape}"
        )

    def test_tensor_channel_dim(self, grey_png):
        """Input must be 3-channel (RGB) after PIL conversion."""
        tensor, _ = preprocess_xray(grey_png, image_size=224)

        assert tensor.shape[1] == 3, (
            f"Expected 3 channels, got tensor.shape={tensor.shape}"
        )

    def test_tensor_spatial_dims(self, grey_png):
        """Spatial dimensions must match image_size after Resize."""
        tensor, _ = preprocess_xray(grey_png, image_size=224)

        assert tensor.shape[2] == 224 and tensor.shape[3] == 224, (
            f"Expected 224x224, got tensor.shape={tensor.shape}"
        )

    def test_tensor_full_shape(self, grey_png):
        """Convenience test: assert the full shape tuple at once."""
        tensor, _ = preprocess_xray(grey_png, image_size=224)

        assert tensor.shape == torch.Size([1, 3, 224, 224]), (
            f"Unexpected tensor shape: {tensor.shape}"
        )

    def test_original_array_shape(self, grey_png):
        """
        original_np must be (image_size, image_size, 3) -- the display copy
        used for heatmap overlay.  Channels-last HxWxC is required by cv2
        and matplotlib.
        """
        _, original_np = preprocess_xray(grey_png, image_size=224)

        assert original_np.shape == (224, 224, 3), (
            f"Unexpected original_np shape: {original_np.shape}"
        )

    def test_original_array_dtype_is_uint8(self, grey_png):
        """
        original_np must be uint8 so cv2.addWeighted and matplotlib.imshow
        handle it correctly without extra casting.
        """
        _, original_np = preprocess_xray(grey_png, image_size=224)

        assert original_np.dtype == np.uint8, (
            f"original_np.dtype={original_np.dtype}, expected uint8"
        )

    def test_custom_image_size_respected(self, grey_png):
        """Passing image_size=64 must produce (1, 3, 64, 64) and (64, 64, 3)."""
        tensor, original_np = preprocess_xray(grey_png, image_size=64)

        assert tensor.shape == torch.Size([1, 3, 64, 64]), (
            f"With image_size=64, expected (1,3,64,64), got {tensor.shape}"
        )
        assert original_np.shape == (64, 64, 3), (
            f"With image_size=64, expected (64,64,3), got {original_np.shape}"
        )

    def test_greyscale_image_converted_to_rgb(self, tmp_path):
        """
        A greyscale (mode 'L') PNG must be accepted and its tensor must
        still have 3 channels after .convert('RGB') inside preprocess_xray().
        """
        grey_l_path = str(tmp_path / "grey_L.png")
        Image.new("L", (64, 64), color=128).save(grey_l_path)

        tensor, original_np = preprocess_xray(grey_l_path, image_size=64)

        assert tensor.shape[1] == 3, (
            "Greyscale image was not converted to 3-channel RGB"
        )
        assert original_np.shape[2] == 3


# ---------------------------------------------------------------------------
# Test 4 -- normalisation shifts values outside [0, 1]
# ---------------------------------------------------------------------------

class TestPreprocessXrayNormalized:
    """
    ImageNet Normalize() subtracts the channel mean and divides by std.
    For an input with uniform pixel values the output must contain values
    both below 0 and above 0, with at least the minimum being negative.

    Why?
    ----
    ImageNet mean is approximately [0.485, 0.456, 0.406].
    ToTensor maps uint8 [0, 255] to float [0.0, 1.0], so a grey (128, 128, 128)
    image becomes [0.502, 0.502, 0.502] per channel.
    After normalisation:
        R channel: (0.502 - 0.485) / 0.229 =  +0.074
        G channel: (0.502 - 0.456) / 0.224 =  +0.205
        B channel: (0.502 - 0.406) / 0.225 =  +0.427
    All positive for grey(128).  But a darker image (grey 50) gives:
        R channel: (0.196 - 0.485) / 0.229 = -1.262  (negative!)
    So we use a darker image here to guarantee at least one negative value.
    """

    @pytest.fixture
    def dark_png(self, tmp_path):
        """Very dark grey (10, 10, 10) PNG -- guaranteed to produce negative tensor values."""
        path = str(tmp_path / "dark.png")
        Image.new("RGB", (256, 256), color=(10, 10, 10)).save(path)
        return path

    def test_tensor_min_is_negative(self, dark_png):
        """
        After ImageNet normalisation, at least one pixel value must be
        negative.  A non-negative minimum would mean Normalize was skipped
        or the wrong constants were used.
        """
        tensor, _ = preprocess_xray(dark_png, image_size=224)

        assert tensor.min().item() < 0.0, (
            f"tensor.min()={tensor.min().item():.4f} is not negative. "
            "Did preprocess_xray() apply Normalize correctly?"
        )

    def test_tensor_is_not_bounded_0_to_1(self, dark_png):
        """
        After normalisation, values should NOT all lie in [0, 1].
        This is the inverse of what ToTensor alone would produce, and
        confirms that the Normalize step is actually doing work.
        """
        tensor, _ = preprocess_xray(dark_png, image_size=224)

        still_in_unit_range = (tensor.min().item() >= 0.0
                               and tensor.max().item() <= 1.0)
        assert not still_in_unit_range, (
            "Tensor values are still in [0, 1] after normalisation -- "
            "Normalize may not have been applied"
        )

    def test_tensor_dtype_is_float32(self, grey_png):
        """
        ToTensor + Normalize must produce a float32 tensor.  float64 would
        double memory usage and cause type-mismatch errors in DenseNet's
        BatchNorm layers.
        """
        tensor, _ = preprocess_xray(grey_png, image_size=224)

        assert tensor.dtype == torch.float32, (
            f"Expected float32 tensor, got {tensor.dtype}"
        )

    def test_original_array_unaffected_by_normalisation(self, grey_png):
        """
        original_np is captured before Normalize runs.  Its values must
        still be in [0, 255] -- confirming that normalisation operates on
        the tensor copy, not the numpy array.
        """
        _, original_np = preprocess_xray(grey_png, image_size=224)

        assert int(original_np.min()) >= 0, "original_np has negative values"
        assert int(original_np.max()) <= 255, "original_np has values above 255"


# ---------------------------------------------------------------------------
# Test 5 -- save_visualization writes a valid PNG
# ---------------------------------------------------------------------------

class TestSaveVisualization:
    """
    save_visualization() must create a three-panel PNG on disk.
    No model or GradCAM object is required -- we pass synthetic uint8 arrays
    directly, which is all the function needs.
    """

    @pytest.fixture
    def panels(self):
        """Three synthetic (224, 224, 3) uint8 arrays for the three panels."""
        rng = np.random.default_rng(seed=99)
        make = lambda: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        return make(), make(), make()   # original, overlaid, heatmap

    def test_output_file_is_created(self, panels, tmp_path):
        """save_visualization() must write a file at save_path."""
        original, overlaid, heatmap = panels
        out = str(tmp_path / "result.png")
        save_visualization(original, overlaid, heatmap, "Normal", 0.92, out)
        assert Path(out).exists(), "save_visualization() did not create the output file"

    def test_output_is_a_valid_png(self, panels, tmp_path):
        """The written file must be readable as a PNG by PIL."""
        original, overlaid, heatmap = panels
        out = str(tmp_path / "result.png")
        save_visualization(original, overlaid, heatmap, "Pneumonia", 0.87, out)

        img = Image.open(out)
        assert img.format == "PNG", f"Expected PNG, got {img.format}"

    def test_creates_missing_parent_dirs(self, panels, tmp_path):
        """
        If the parent directory does not exist, save_visualization() must
        create it automatically (via Path.mkdir(parents=True, exist_ok=True)).
        """
        original, overlaid, heatmap = panels
        deep = str(tmp_path / "a" / "b" / "c" / "result.png")
        save_visualization(original, overlaid, heatmap, "Normal", 0.75, deep)
        assert Path(deep).exists()

    def test_pneumonia_label_does_not_raise(self, panels, tmp_path):
        """
        Passing prediction_label='Pneumonia' must not raise -- this exercises
        the red (#d63c3c) title-colour branch of the function.
        """
        original, overlaid, heatmap = panels
        out = str(tmp_path / "pneumonia_result.png")
        save_visualization(original, overlaid, heatmap, "Pneumonia", 0.94, out)
        assert Path(out).exists()

    def test_output_has_nonzero_file_size(self, panels, tmp_path):
        """
        A non-zero file size confirms that matplotlib actually wrote pixel
        data and didn't produce an empty file (which can happen if plt.savefig
        is called before fig has been populated).
        """
        original, overlaid, heatmap = panels
        out = str(tmp_path / "result.png")
        save_visualization(original, overlaid, heatmap, "Normal", 0.60, out)
        assert Path(out).stat().st_size > 1024, (
            "Output file is suspiciously small -- matplotlib may not have "
            "rendered any content"
        )


# ---------------------------------------------------------------------------
# Test 6 -- generate_batch_report runs the full pipeline end-to-end
# ---------------------------------------------------------------------------

class TestGenerateBatchReport:
    """
    generate_batch_report() chains preprocess -> gradcam -> predict ->
    overlay -> save for every image in a list.

    We use the TinyCNN fixture from conftest.py (auto-discovered by pytest
    from tests/conftest.py) to avoid needing a real DenseNet checkpoint.
    A synthetic grey PNG written to tmp_path stands in for a real X-ray.
    """

    def test_returns_list_with_one_entry_per_image(
        self, tiny_model, grey_png, tmp_path
    ):
        """
        The return value must be a list with exactly len(image_paths) entries.
        """
        from src.gradcam import GradCAM

        model, target_layer = tiny_model
        with GradCAM(model, target_layer) as cam:
            results = generate_batch_report(
                image_paths=[grey_png],
                model=model,
                gradcam=cam,
                output_dir=str(tmp_path / "batch"),
            )

        assert isinstance(results, list), "generate_batch_report() must return a list"
        assert len(results) == 1, (
            f"Expected 1 result for 1 image, got {len(results)}"
        )

    def test_result_dicts_have_required_keys(self, tiny_model, grey_png, tmp_path):
        """
        Each dict in the returned list must have all four required keys.
        """
        from src.gradcam import GradCAM

        model, target_layer = tiny_model
        with GradCAM(model, target_layer) as cam:
            results = generate_batch_report(
                image_paths=[grey_png],
                model=model,
                gradcam=cam,
                output_dir=str(tmp_path / "batch2"),
            )

        required = {"image", "prediction", "confidence", "output_path"}
        assert required.issubset(results[0].keys()), (
            f"Missing keys: {required - set(results[0].keys())}"
        )

    def test_output_png_is_created_on_disk(self, tiny_model, grey_png, tmp_path):
        """
        generate_batch_report() must write a PNG file to output_dir for
        each processed image.
        """
        from src.gradcam import GradCAM
        from pathlib import Path as _P

        model, target_layer = tiny_model
        out_dir = tmp_path / "batch3"
        with GradCAM(model, target_layer) as cam:
            results = generate_batch_report(
                image_paths=[grey_png],
                model=model,
                gradcam=cam,
                output_dir=str(out_dir),
            )

        out_path = _P(results[0]["output_path"])
        assert out_path.exists(), (
            f"Expected output file at {out_path} but it was not created"
        )

    def test_output_filename_uses_stem_plus_gradcam(
        self, tiny_model, grey_png, tmp_path
    ):
        """
        The output file must be named {original_stem}_gradcam.png.
        e.g. 'fake_xray.png' -> 'fake_xray_gradcam.png'.
        """
        from src.gradcam import GradCAM
        from pathlib import Path as _P

        model, target_layer = tiny_model
        with GradCAM(model, target_layer) as cam:
            results = generate_batch_report(
                image_paths=[grey_png],
                model=model,
                gradcam=cam,
                output_dir=str(tmp_path / "batch4"),
            )

        expected_stem = _P(grey_png).stem + "_gradcam"
        actual_stem   = _P(results[0]["output_path"]).stem
        assert actual_stem == expected_stem, (
            f"Expected filename stem '{expected_stem}', got '{actual_stem}'"
        )

    def test_confidence_is_in_unit_range(self, tiny_model, grey_png, tmp_path):
        """
        The confidence value in each result dict must be a valid softmax
        probability in [0, 1].
        """
        from src.gradcam import GradCAM

        model, target_layer = tiny_model
        with GradCAM(model, target_layer) as cam:
            results = generate_batch_report(
                image_paths=[grey_png],
                model=model,
                gradcam=cam,
                output_dir=str(tmp_path / "batch5"),
            )

        conf = results[0]["confidence"]
        assert 0.0 <= conf <= 1.0, (
            f"confidence={conf} is outside [0, 1]"
        )
