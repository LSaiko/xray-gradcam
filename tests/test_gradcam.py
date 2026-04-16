"""
test_gradcam.py -- Unit tests for src/gradcam.py.

Design principles
-----------------
* No real model weights, no real images.
  All tests use the TinyCNN fixture from conftest.py and torch.randn() tensors.
  This means the test suite runs in milliseconds and works fully offline.

* Each test exercises one behavioural contract.
  We avoid testing implementation details (hook internals, gradient values)
  and focus on the public interface: output shape, value range, index contract,
  and lifecycle safety (hooks cleaned up, context manager exits cleanly).

* Deterministic where it matters.
  We seed torch.manual_seed() in tests that rely on the output not being
  all-zero (a degenerate case that can occur with unlucky random weights
  mapping every activation to exactly zero after ReLU in Grad-CAM).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# sys.path guard -- makes 'src.*' importable when pytest is run from the
# project root OR from the tests/ directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gradcam import GradCAM, GradCAMPlusPlus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_input(seed=0, batch=1, channels=3, h=32, w=32):
    """
    Return a reproducible random input tensor of shape (batch, C, H, W).

    Using a fixed seed guarantees that 'unlucky' random weights don't produce
    an all-zero CAM (which would cause the max() > 0 assertions to fail).
    We use a separate helper rather than module-level state so each test can
    request its own seed without affecting others.
    """
    torch.manual_seed(seed)
    return torch.randn(batch, channels, h, w)


# ---------------------------------------------------------------------------
# Test 1 -- output shape and value range (GradCAM)
# ---------------------------------------------------------------------------

class TestGradCAMOutputShape:
    """GradCAM.generate() must return a 2-D float array in [0, 1]."""

    def test_cam_is_2d(self, tiny_model):
        """The CAM array should have exactly two dimensions (H_feat, W_feat)."""
        model, target_layer = tiny_model
        x = make_input(seed=1)

        with GradCAM(model, target_layer) as cam:
            cam_array, _ = cam.generate(x)

        # A 2-D array means the spatial dims were preserved and the batch /
        # channel dims were correctly reduced by the weighted-sum step.
        assert cam_array.ndim == 2, (
            f"Expected 2-D CAM, got shape {cam_array.shape}"
        )

    def test_cam_values_in_unit_range(self, tiny_model):
        """Every pixel in the CAM must lie in [0.0, 1.0] after normalisation."""
        model, target_layer = tiny_model
        x = make_input(seed=2)

        with GradCAM(model, target_layer) as cam:
            cam_array, _ = cam.generate(x)

        # min-max normalisation in generate() guarantees this; if it regresses
        # (e.g. epsilon removed) values can exceed 1 or go negative.
        assert float(cam_array.min()) >= 0.0, (
            f"CAM has negative values (min={cam_array.min():.6f})"
        )
        assert float(cam_array.max()) <= 1.0, (
            f"CAM exceeds 1.0 (max={cam_array.max():.6f})"
        )

    def test_cam_is_not_all_zeros(self, tiny_model):
        """
        A non-trivial input should produce at least one non-zero activation.

        An all-zero CAM means either:
        a) Every gradient was zero (vanishing gradient / dead ReLU bug), or
        b) The normalisation divided by zero and produced NaN/0.
        Either case is a bug in generate().
        """
        model, target_layer = tiny_model
        x = make_input(seed=3)

        with GradCAM(model, target_layer) as cam:
            cam_array, _ = cam.generate(x)

        assert float(cam_array.max()) > 0.0, (
            "CAM is all zeros -- check ReLU or normalisation in generate()"
        )

    def test_cam_dtype_is_float32(self, tiny_model):
        """generate() should return a float32 numpy array (not float64)."""
        model, target_layer = tiny_model
        x = make_input(seed=4)

        with GradCAM(model, target_layer) as cam:
            cam_array, _ = cam.generate(x)

        assert isinstance(cam_array, np.ndarray), "CAM should be a numpy array"
        assert cam_array.dtype == np.float32, (
            f"Expected float32, got {cam_array.dtype}"
        )


# ---------------------------------------------------------------------------
# Test 2 -- hook cleanup
# ---------------------------------------------------------------------------

class TestGradCAMRemovesHooks:
    """remove_hooks() must run without raising and must not crash on a second call."""

    def test_remove_hooks_does_not_raise(self, tiny_model):
        """
        Calling remove_hooks() on a freshly constructed GradCAM must not raise.

        We cannot easily introspect PyTorch's internal hook registry from
        outside the module, but we *can* assert that the public method
        completes without error -- which would catch the most common mistake
        of storing the wrong handle type or calling .remove() twice.
        """
        model, target_layer = tiny_model
        cam = GradCAM(model, target_layer)

        # This must not raise AttributeError, TypeError, or RuntimeError.
        cam.remove_hooks()

    def test_handle_attributes_exist_before_removal(self, tiny_model):
        """
        The two handle attributes must be set by __init__ so that
        remove_hooks() always has something to call .remove() on.

        If __init__ forgets to store a handle, remove_hooks() would raise
        AttributeError -- which would be caught by the test above but this
        test makes the *source* of the failure explicit.
        """
        model, target_layer = tiny_model
        cam = GradCAM(model, target_layer)

        assert hasattr(cam, "_forward_handle"), (
            "GradCAM.__init__ must store the forward hook handle as "
            "self._forward_handle"
        )
        assert hasattr(cam, "_backward_handle"), (
            "GradCAM.__init__ must store the backward hook handle as "
            "self._backward_handle"
        )

        cam.remove_hooks()  # clean up after inspection


# ---------------------------------------------------------------------------
# Test 3 -- context manager
# ---------------------------------------------------------------------------

class TestGradCAMContextManager:
    """GradCAM must be usable as a context manager; hooks cleaned up on exit."""

    def test_generate_inside_with_block_does_not_raise(self, tiny_model):
        """
        The typical usage pattern is 'with GradCAM(...) as cam: cam.generate()'.
        This must complete without any exception under normal conditions.
        """
        model, target_layer = tiny_model
        x = make_input(seed=5)

        # pytest will fail this test if *any* exception escapes the with block.
        with GradCAM(model, target_layer) as cam:
            cam_array, class_idx = cam.generate(x)

        # Basic sanity on the results -- confirms generate() actually ran.
        assert cam_array is not None
        assert isinstance(class_idx, int)

    def test_context_manager_cleans_up_on_exception(self, tiny_model):
        """
        __exit__ must be called (and hooks removed) even when an exception
        is raised inside the with block.

        We simulate a mid-block failure and confirm that:
        1. The exception propagates correctly (not swallowed by __exit__).
        2. remove_hooks() was called -- verified indirectly by patching.
        """
        from unittest.mock import patch

        model, target_layer = tiny_model
        cam_obj = GradCAM(model, target_layer)

        with patch.object(cam_obj, "remove_hooks", wraps=cam_obj.remove_hooks) as mock_remove:
            with pytest.raises(RuntimeError, match="simulated failure"):
                with cam_obj:
                    raise RuntimeError("simulated failure")

            # __exit__ must have called remove_hooks() exactly once.
            mock_remove.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4 -- class_idx auto-selection
# ---------------------------------------------------------------------------

class TestGradCAMClassIdxReturned:
    """When class_idx=None, generate() must return a valid class index."""

    def test_returned_class_idx_is_valid(self, tiny_model):
        """
        With a 2-class model and class_idx=None, the returned index must
        be either 0 or 1.

        This checks that generate() correctly calls argmax() on the logits
        rather than returning a hardcoded value or the raw float.
        """
        model, target_layer = tiny_model
        x = make_input(seed=6)

        with GradCAM(model, target_layer) as cam:
            _, returned_idx = cam.generate(x, class_idx=None)

        assert returned_idx in (0, 1), (
            f"Expected class index 0 or 1 for a 2-class model, got {returned_idx}"
        )

    def test_returned_class_idx_is_int(self, tiny_model):
        """
        The returned class index must be a plain Python int, not a
        torch.Tensor or numpy integer, so it is JSON-serialisable and
        can be used as a list index without explicit casting.
        """
        model, target_layer = tiny_model
        x = make_input(seed=7)

        with GradCAM(model, target_layer) as cam:
            _, returned_idx = cam.generate(x, class_idx=None)

        assert isinstance(returned_idx, int), (
            f"class_idx should be a plain int, got {type(returned_idx)}"
        )


# ---------------------------------------------------------------------------
# Test 5 -- explicit class_idx is respected
# ---------------------------------------------------------------------------

class TestGradCAMRespectsExplicitClassIdx:
    """Passing class_idx=N must produce a CAM for class N, not the top class."""

    def test_explicit_class_idx_0_is_returned(self, tiny_model):
        """
        When class_idx=0 is passed, the *returned* class_idx must be 0.

        This confirms that generate() passes the argument through to the
        backward call (logits[0, 0].backward()) rather than silently
        overriding it with argmax.
        """
        model, target_layer = tiny_model
        x = make_input(seed=8)

        with GradCAM(model, target_layer) as cam:
            _, returned_idx = cam.generate(x, class_idx=0)

        assert returned_idx == 0, (
            f"Passed class_idx=0 but generate() returned {returned_idx}"
        )

    def test_explicit_class_idx_1_is_returned(self, tiny_model):
        """
        When class_idx=1 is passed, the returned class_idx must be 1.

        Combined with the test above, these two tests confirm that generate()
        uses the argument verbatim and does not always clamp to 0 or always
        call argmax.
        """
        model, target_layer = tiny_model
        x = make_input(seed=9)

        with GradCAM(model, target_layer) as cam:
            _, returned_idx = cam.generate(x, class_idx=1)

        assert returned_idx == 1, (
            f"Passed class_idx=1 but generate() returned {returned_idx}"
        )

    def test_explicit_idx_produces_different_cam_than_argmax(self, tiny_model):
        """
        Explaining class 0 versus class 1 should (in general) produce
        different heatmaps, because different neurons drive each class.

        This is a probabilistic test: with random weights the two CAMs are
        almost certainly different.  We use a fixed seed to guarantee it.

        If this test flakes, it means the network happened to produce
        identical gradients for both classes -- extremely unlikely but
        possible with a specific seed.  The fallback is to change the seed.
        """
        model, target_layer = tiny_model
        x = make_input(seed=42)

        with GradCAM(model, target_layer) as cam_0:
            cam_array_0, _ = cam_0.generate(x, class_idx=0)

        # Re-create GradCAM to get fresh hooks; the previous context manager
        # already called remove_hooks() on exit.
        with GradCAM(model, target_layer) as cam_1:
            cam_array_1, _ = cam_1.generate(x, class_idx=1)

        # np.allclose returns True if arrays are element-wise almost equal.
        # We assert they are NOT identical -- different classes, different CAMs.
        assert not np.allclose(cam_array_0, cam_array_1), (
            "CAMs for class 0 and class 1 are identical -- "
            "generate() may be ignoring the class_idx argument"
        )


# ---------------------------------------------------------------------------
# Test 6 -- GradCAMPlusPlus output shape and range
# ---------------------------------------------------------------------------

class TestGradCAMPlusPlusOutputShape:
    """GradCAMPlusPlus must satisfy the same interface contract as GradCAM."""

    def test_plusplus_cam_is_2d(self, tiny_model):
        """GradCAMPlusPlus.generate() must return a 2-D array."""
        model, target_layer = tiny_model
        x = make_input(seed=10)

        with GradCAMPlusPlus(model, target_layer) as cam:
            cam_array, _ = cam.generate(x)

        assert cam_array.ndim == 2, (
            f"Expected 2-D CAM from GradCAMPlusPlus, got shape {cam_array.shape}"
        )

    def test_plusplus_values_in_unit_range(self, tiny_model):
        """GradCAMPlusPlus values must be in [0.0, 1.0]."""
        model, target_layer = tiny_model
        x = make_input(seed=11)

        with GradCAMPlusPlus(model, target_layer) as cam:
            cam_array, _ = cam.generate(x)

        assert float(cam_array.min()) >= 0.0
        assert float(cam_array.max()) <= 1.0

    def test_plusplus_cam_is_not_all_zeros(self, tiny_model):
        """GradCAMPlusPlus must produce a non-trivial heatmap."""
        model, target_layer = tiny_model
        x = make_input(seed=12)

        with GradCAMPlusPlus(model, target_layer) as cam:
            cam_array, _ = cam.generate(x)

        assert float(cam_array.max()) > 0.0, (
            "GradCAMPlusPlus CAM is all zeros -- "
            "check the second-order weight formula in generate()"
        )

    def test_plusplus_dtype_is_float32(self, tiny_model):
        """GradCAMPlusPlus must return a float32 numpy array."""
        model, target_layer = tiny_model
        x = make_input(seed=13)

        with GradCAMPlusPlus(model, target_layer) as cam:
            cam_array, _ = cam.generate(x)

        assert isinstance(cam_array, np.ndarray)
        assert cam_array.dtype == np.float32

    def test_plusplus_shape_matches_gradcam_shape(self, tiny_model):
        """
        GradCAM and GradCAMPlusPlus must produce arrays of the same shape
        for the same input, because the spatial dimensions come from the
        same target layer and the same input tensor.

        If this fails, GradCAMPlusPlus is likely collapsing or broadcasting
        dimensions incorrectly in the second-order weight computation.
        """
        model, target_layer = tiny_model
        x = make_input(seed=14)

        with GradCAM(model, target_layer) as cam:
            cam_v1, _ = cam.generate(x)

        with GradCAMPlusPlus(model, target_layer) as cam:
            cam_v2, _ = cam.generate(x)

        assert cam_v1.shape == cam_v2.shape, (
            f"Shape mismatch: GradCAM={cam_v1.shape}, "
            f"GradCAMPlusPlus={cam_v2.shape}"
        )
