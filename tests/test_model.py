"""
test_model.py -- Unit tests for src/model.py.

Design principles
-----------------
* load_model() is tested by saving a real DenseNet121 state-dict to a pytest
  tmp_path directory.  No internet access is required because pretrained=False.
  A class-scoped fixture builds and saves the checkpoint once for the whole
  TestLoadModel class, avoiding repeated 30 MB writes.

* test_get_target_layer is the one test that builds a real DenseNet121
  (pretrained=False).  This is unavoidable because get_target_layer() accesses
  a specific nested attribute that only exists on that architecture.
  pretrained=False means no internet access -- the weights are random but the
  architecture is fully constructed.

* Determinism: torch.manual_seed() is called in every test that checks
  numerical ranges, so random weight initialisation does not produce
  degenerate outputs (e.g. logits that are exactly equal, giving
  confidence = 0.5 for both classes and an ambiguous max() index).
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# sys.path guard
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import get_target_layer, load_model, predict


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_flat_model(num_classes=2):
    """
    Return a tiny nn.Sequential that accepts a [1, 3, 224, 224] tensor and
    emits [1, num_classes] logits.

    We use AdaptiveAvgPool2d + Flatten instead of a bare Linear so that the
    model accepts the same 4-D input shape that predict() expects, mirroring
    the real DenseNet121 pipeline without any of its complexity.

    Architecture
    ------------
    AdaptiveAvgPool2d(1)   # (1, 3, 224, 224) -> (1, 3, 1, 1)
    Flatten                # -> (1, 3)
    Linear(3, num_classes) # -> (1, num_classes)
    """
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, num_classes),
    ).eval()


# ---------------------------------------------------------------------------
# Test 1 -- output keys
# ---------------------------------------------------------------------------

class TestPredictOutputKeys:
    """predict() must always return a dict with exactly the three required keys."""

    def test_result_has_predicted_class_key(self):
        torch.manual_seed(0)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert "predicted_class" in result, (
            "predict() result is missing the 'predicted_class' key"
        )

    def test_result_has_confidence_key(self):
        torch.manual_seed(1)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert "confidence" in result, (
            "predict() result is missing the 'confidence' key"
        )

    def test_result_has_probabilities_key(self):
        torch.manual_seed(2)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert "probabilities" in result, (
            "predict() result is missing the 'probabilities' key"
        )

    def test_result_has_no_extra_keys(self):
        """
        The contract is exactly three keys.  Extra keys could indicate a
        refactor that changed the return schema without updating callers.
        """
        torch.manual_seed(3)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        expected_keys = {"predicted_class", "confidence", "probabilities"}
        assert set(result.keys()) == expected_keys, (
            f"predict() returned unexpected keys: "
            f"{set(result.keys()) - expected_keys}"
        )

    def test_probabilities_is_a_list(self):
        """
        probabilities must be a plain Python list so it is JSON-serialisable
        and can be used as a list index without extra casting.
        """
        torch.manual_seed(4)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert isinstance(result["probabilities"], list), (
            f"'probabilities' should be a list, got {type(result['probabilities'])}"
        )

    def test_confidence_is_a_float(self):
        """
        confidence must be a plain Python float, not a torch.Tensor or
        numpy scalar, so callers can format it directly (f'{conf:.1%}').
        """
        torch.manual_seed(5)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert isinstance(result["confidence"], float), (
            f"'confidence' should be a float, got {type(result['confidence'])}"
        )

    def test_predicted_class_is_a_string(self):
        torch.manual_seed(6)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert isinstance(result["predicted_class"], str), (
            f"'predicted_class' should be a str, "
            f"got {type(result['predicted_class'])}"
        )


# ---------------------------------------------------------------------------
# Test 2 -- confidence range and probability sum
# ---------------------------------------------------------------------------

class TestPredictConfidenceRange:
    """Softmax outputs must be valid probabilities: each in [0,1], sum to 1."""

    def test_confidence_at_least_zero(self):
        torch.manual_seed(10)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert result["confidence"] >= 0.0, (
            f"confidence={result['confidence']} is negative -- "
            "softmax output should never be < 0"
        )

    def test_confidence_at_most_one(self):
        torch.manual_seed(11)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert result["confidence"] <= 1.0, (
            f"confidence={result['confidence']} exceeds 1.0 -- "
            "softmax output should never be > 1"
        )

    def test_probabilities_sum_to_one(self):
        """
        All softmax probabilities must sum to 1.0 within floating-point
        tolerance.  A sum far from 1 would indicate log_softmax was used
        instead of softmax, or that the output was not normalised at all.
        """
        torch.manual_seed(12)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        total = sum(result["probabilities"])
        assert abs(total - 1.0) < 1e-4, (
            f"probabilities sum to {total:.6f}, expected ~1.0 "
            "(is softmax applied correctly?)"
        )

    def test_probabilities_length_matches_num_classes(self):
        """
        The length of the probabilities list must equal the model's number
        of output classes.  A length mismatch means the squeeze/tolist
        step in predict() is dropping or duplicating values.
        """
        torch.manual_seed(13)
        num_classes = 2
        model = _make_flat_model(num_classes=num_classes)
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert len(result["probabilities"]) == num_classes, (
            f"Expected {num_classes} probabilities, "
            f"got {len(result['probabilities'])}"
        )

    def test_confidence_equals_max_probability(self):
        """
        confidence must equal the largest value in probabilities, because
        it is defined as the softmax score of the predicted class (the
        argmax class).
        """
        torch.manual_seed(14)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert abs(result["confidence"] - max(result["probabilities"])) < 1e-6, (
            "confidence should equal max(probabilities) -- "
            "they appear to be out of sync"
        )

    def test_all_individual_probabilities_in_unit_range(self):
        """Every element of probabilities (not just confidence) must be in [0,1]."""
        torch.manual_seed(15)
        model = _make_flat_model()
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        for i, p in enumerate(result["probabilities"]):
            assert 0.0 <= p <= 1.0, (
                f"probabilities[{i}]={p:.6f} is outside [0, 1]"
            )


# ---------------------------------------------------------------------------
# Test 3 -- predicted class name
# ---------------------------------------------------------------------------

class TestPredictClassIsValid:
    """predicted_class must be one of the known class names."""

    def test_predicted_class_in_default_names(self):
        """
        With the default class_names=['Normal', 'Pneumonia'], the predicted
        class string must be exactly one of those two values.

        This guards against off-by-one indexing in argmax or a mismatch
        between class_names length and num_classes.
        """
        torch.manual_seed(20)
        model = _make_flat_model(num_classes=2)
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu")

        assert result["predicted_class"] in ("Normal", "Pneumonia"), (
            f"predicted_class='{result['predicted_class']}' is not one of "
            "the expected class names ['Normal', 'Pneumonia']"
        )

    def test_predicted_class_respects_custom_names(self):
        """
        When custom class_names are supplied, the returned string must come
        from that list -- not the hard-coded defaults.
        """
        torch.manual_seed(21)
        custom_names = ["Healthy", "Infected", "Uncertain"]
        model = _make_flat_model(num_classes=3)
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu", class_names=custom_names)

        assert result["predicted_class"] in custom_names, (
            f"predicted_class='{result['predicted_class']}' is not in "
            f"custom class_names={custom_names}"
        )

    def test_predicted_class_consistent_with_confidence(self):
        """
        The confidence must equal probabilities[idx] where idx is the index
        of predicted_class in class_names.  This cross-checks that the
        label and the score refer to the same class.
        """
        torch.manual_seed(22)
        class_names = ["Normal", "Pneumonia"]
        model = _make_flat_model(num_classes=2)
        x = torch.randn(1, 3, 224, 224)
        result = predict(model, x, device="cpu", class_names=class_names)

        idx = class_names.index(result["predicted_class"])
        assert abs(result["confidence"] - result["probabilities"][idx]) < 1e-6, (
            f"confidence ({result['confidence']:.6f}) does not match "
            f"probabilities[{idx}] ({result['probabilities'][idx]:.6f}) "
            "for predicted_class='{result['predicted_class']}'"
        )

    def test_predict_accepts_3d_input(self):
        """
        predict() must add the batch dim automatically when a 3-D tensor
        (C, H, W) is passed, not raise a shape error.
        """
        torch.manual_seed(23)
        model = _make_flat_model(num_classes=2)
        # 3-D input: no batch dimension
        x = torch.randn(3, 224, 224)
        result = predict(model, x, device="cpu")

        assert result["predicted_class"] in ("Normal", "Pneumonia")


# ---------------------------------------------------------------------------
# Test 4 -- get_target_layer returns a Conv2d
# ---------------------------------------------------------------------------

class TestGetTargetLayer:
    """get_target_layer() must return a real nn.Conv2d from DenseNet121."""

    @pytest.fixture(scope="class")
    def densenet(self):
        """
        Build DenseNet121 with pretrained=False once per test class.

        scope='class' means the model is constructed once and shared across
        all methods in this class -- DenseNet121 takes ~0.5 s to build on
        slow hardware, so we avoid rebuilding it four times.
        """
        from torchvision import models
        model = models.densenet121(pretrained=False)
        model.eval()
        return model

    def test_returns_conv2d_instance(self, densenet):
        """The target layer must be an nn.Conv2d (not BatchNorm, ReLU, etc.)."""
        layer = get_target_layer(densenet)

        assert isinstance(layer, nn.Conv2d), (
            f"get_target_layer() returned {type(layer).__name__}, "
            "expected nn.Conv2d.  "
            "If the target path was changed, update get_target_layer() "
            "and this test together."
        )

    def test_layer_is_from_denseblock4(self, densenet):
        """
        The returned layer must be the same object as
        densenet.features.denseblock4.denselayer16.conv2 -- identity check,
        not just type check.

        This guards against get_target_layer() accidentally returning a
        *different* Conv2d from an earlier dense block.
        """
        expected = densenet.features.denseblock4.denselayer16.conv2
        actual   = get_target_layer(densenet)

        assert actual is expected, (
            "get_target_layer() returned a different layer than "
            "model.features.denseblock4.denselayer16.conv2"
        )

    def test_layer_has_expected_out_channels(self, densenet):
        """
        denselayer16.conv2 in DenseNet121 has 32 output channels (the growth
        rate k=32).  If this changes, the Grad-CAM feature maps will have a
        different number of channels and the weighted-sum step will silently
        produce wrong heatmaps.
        """
        layer = get_target_layer(densenet)

        assert layer.out_channels == 32, (
            f"Expected 32 output channels (DenseNet121 growth rate), "
            f"got {layer.out_channels}"
        )

    def test_layer_accepts_grad_hooks(self, densenet):
        """
        The returned layer must support register_forward_hook and
        register_full_backward_hook without raising.  These are the two
        calls that GradCAM.__init__ makes on target_layer.
        """
        layer = get_target_layer(densenet)

        fwd_handle = layer.register_forward_hook(lambda m, i, o: None)
        bwd_handle = layer.register_full_backward_hook(lambda m, gi, go: None)

        # Clean up immediately -- we only care that the calls didn't raise.
        fwd_handle.remove()
        bwd_handle.remove()


# ---------------------------------------------------------------------------
# Test 5 -- load_model() reads a checkpoint and returns an eval-mode model
# ---------------------------------------------------------------------------

class TestLoadModel:
    """
    load_model() must build DenseNet121, replace the classifier head, load
    weights from disk, and return an eval-mode model on the requested device.

    We save a real DenseNet121 state-dict to pytest's tmp_path once per class
    (scope='class') to avoid writing a 30 MB file for every individual test.
    """

    @pytest.fixture(scope="class")
    def bare_checkpoint(self, tmp_path_factory):
        """
        Build DenseNet121(pretrained=False), replace its head with Linear(1024,2),
        and torch.save() the bare state-dict to a temp file.

        scope='class' means this fixture runs once for all methods in
        TestLoadModel, avoiding redundant DenseNet constructions and disk writes.
        """
        import warnings
        from torchvision import models

        # Silence the torchvision 'pretrained is deprecated' warning -- we know,
        # we are intentionally using pretrained=False.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = models.densenet121(pretrained=False)

        net.classifier = nn.Linear(1024, 2)
        ckpt_dir  = tmp_path_factory.mktemp("bare_ckpt")
        ckpt_path = ckpt_dir / "bare_weights.pt"
        torch.save(net.state_dict(), str(ckpt_path))
        return str(ckpt_path)

    @pytest.fixture(scope="class")
    def wrapped_checkpoint(self, tmp_path_factory):
        """
        Same weights saved as a Lightning-style wrapper dict:
        {'state_dict': ..., 'epoch': 10}
        """
        import warnings
        from torchvision import models

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = models.densenet121(pretrained=False)

        net.classifier = nn.Linear(1024, 2)
        ckpt_dir  = tmp_path_factory.mktemp("wrapped_ckpt")
        ckpt_path = ckpt_dir / "wrapped_weights.pt"
        torch.save({"state_dict": net.state_dict(), "epoch": 10}, str(ckpt_path))
        return str(ckpt_path)

    def test_returns_model_in_eval_mode(self, bare_checkpoint):
        """
        load_model() must call model.eval() before returning.
        model.training is True by default after construction; eval() sets it
        to False.  If this flag is True, BatchNorm uses per-batch stats during
        inference, making predictions non-deterministic.
        """
        model = load_model(bare_checkpoint, num_classes=2, device="cpu")
        assert not model.training, (
            "load_model() must call model.eval() -- model.training is True"
        )

    def test_classifier_out_features_matches_num_classes(self, bare_checkpoint):
        """
        The loaded model's final Linear layer must have exactly num_classes
        output neurons.  If the head was not replaced before loading the
        state-dict, this would be 1000 (ImageNet default).
        """
        model = load_model(bare_checkpoint, num_classes=2, device="cpu")
        assert model.classifier.out_features == 2, (
            f"Expected 2 output neurons, got {model.classifier.out_features}"
        )

    def test_model_is_on_requested_device(self, bare_checkpoint):
        """
        All parameters must live on the device passed to load_model().
        We test CPU here because CUDA is not guaranteed in CI.
        """
        model = load_model(bare_checkpoint, num_classes=2, device="cpu")
        device = next(model.parameters()).device
        assert device.type == "cpu", (
            f"Model is on {device}, expected cpu"
        )

    def test_loads_wrapped_state_dict(self, wrapped_checkpoint):
        """
        load_model() must unpack the {'state_dict': ...} wrapper produced by
        PyTorch Lightning's ModelCheckpoint callback.  If it tries to call
        model.load_state_dict() on the outer dict, it will raise a RuntimeError
        about unexpected keys ('state_dict', 'epoch').
        """
        # If the wrapper format is not handled, this line raises RuntimeError.
        model = load_model(wrapped_checkpoint, num_classes=2, device="cpu")
        assert not model.training

    def test_loaded_model_produces_valid_logits(self, bare_checkpoint):
        """
        A forward pass through the loaded model must produce finite logits
        of shape (1, num_classes).  NaN or Inf would indicate a corrupt
        state-dict or a dtype mismatch during loading.
        """
        model = load_model(bare_checkpoint, num_classes=2, device="cpu")
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (1, 2), (
            f"Expected logits shape (1, 2), got {logits.shape}"
        )
        assert torch.isfinite(logits).all(), (
            "Logits contain NaN or Inf -- possible corrupt state-dict"
        )
