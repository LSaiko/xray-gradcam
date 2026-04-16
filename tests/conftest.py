"""
conftest.py -- Shared pytest fixtures for xray-gradcam tests.

pytest automatically discovers and loads conftest.py before any test module
in the same directory (or any subdirectory).  Fixtures defined here are
available to every test file under tests/ without any explicit import.

Why a separate conftest instead of helpers inside each test file?
-----------------------------------------------------------------
* Avoids duplicating the model-construction boilerplate across
  test_gradcam.py, test_model.py, and test_visualize.py.
* pytest's fixture caching (scope='function' by default) re-creates the
  model fresh for every test, preventing state from leaking between tests.
* The fixture is the single source of truth for the tiny CNN architecture --
  if we ever change it, we only change it here.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that "src.*" imports work when
# pytest is invoked from any working directory (project root, tests/, etc.).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Tiny CNN definition
# ---------------------------------------------------------------------------
# We want the smallest network that still exercises the same code paths as
# DenseNet121:
#   1. A Conv2d layer  -- so GradCAM has a spatial target to hook.
#   2. A non-linearity -- mirrors the activation between dense layers.
#   3. Spatial pooling -- collapses (N, C, H, W) to (N, C) before the head,
#                         just like DenseNet's global average-pool.
#   4. A Linear head   -- produces logits for N classes.
#
# Using a real DenseNet121 in tests would require either:
#   * Downloading 30 MB of pretrained weights (breaks offline CI), or
#   * Instantiating the full architecture (slow, ~7 million parameters).
# This tiny network has <2 000 parameters and constructs in milliseconds.

class TinyCNN(nn.Module):
    """
    Minimal two-class CNN used exclusively in tests.

    Architecture
    ------------
    Input (1, 3, H, W)
        |
        Conv2d(3, 8, kernel_size=3, padding=1)  <-- GradCAM target layer
        ReLU
        AdaptiveAvgPool2d(1)                    <-- collapses spatial dims
        Flatten                                 <-- (N, 8, 1, 1) -> (N, 8)
        Linear(8, num_classes)                  <-- produces logits
    """

    def __init__(self, num_classes=2):
        super().__init__()
        # conv is stored as a named attribute so get_target_layer() in the
        # fixture can return it directly (model.conv), mirroring how
        # model.py's get_target_layer() returns model.features.denseblock4...
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # AdaptiveAvgPool2d(1) reduces any spatial size to (1, 1), so the
        # tests pass tensors of different sizes without shape errors.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.conv(x)        # (N, 8, H, W)
        x = self.relu(x)
        x = self.pool(x)        # (N, 8, 1, 1)
        x = self.flatten(x)     # (N, 8)
        return self.classifier(x)  # (N, num_classes)


# ---------------------------------------------------------------------------
# Return type for the fixture
# ---------------------------------------------------------------------------
# A NamedTuple gives tests clean attribute access (model, target_layer)
# instead of tuple-unpacking magic (model, layer = tiny_model).
class TinyModelFixture(NamedTuple):
    model: nn.Module
    target_layer: nn.Module


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    """
    Return a freshly constructed TinyCNN in eval mode and its GradCAM target.

    Scope: 'function' (default) -- a new model is created for every test
    that requests this fixture.  This prevents any accidental state sharing
    (e.g. gradients left over from a previous test's backward pass).

    Usage in a test::

        def test_something(tiny_model):
            model, target_layer = tiny_model
            ...

    Returns:
        TinyModelFixture(model, target_layer):
            * model        -- TinyCNN instance in eval mode, on CPU.
            * target_layer -- the Conv2d layer (model.conv) that GradCAM
                              should register its hooks on.
    """
    model = TinyCNN(num_classes=2)

    # eval() is mandatory: batch-norm and dropout behave differently in train
    # mode, making outputs non-deterministic.  GradCAM tests need the forward
    # pass to be reproducible across repeated calls within the same test.
    model.eval()

    # model.conv is the only Conv2d in TinyCNN, making it the natural
    # equivalent of DenseNet121's denseblock4.denselayer16.conv2.
    target_layer = model.conv

    return TinyModelFixture(model=model, target_layer=target_layer)
