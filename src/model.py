"""
model.py — DenseNet121 loader and inference utilities for X-ray Grad-CAM.

DenseNet121 is the backbone of choice here because it was the architecture
used in the seminal CheXNet paper (Rajpurkar et al., 2017) for chest X-ray
pathology detection, and its dense connectivity means gradients flow cleanly
back to early feature maps — which is exactly what Grad-CAM needs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def load_model(checkpoint_path, num_classes=2, device="cpu"):
    """
    Build a DenseNet121, swap in a task-specific classifier head, and load
    saved weights from a checkpoint.

    DenseNet121's ImageNet classifier outputs 1000 logits.  For binary
    (Normal / Pneumonia) or any other downstream task we replace that final
    linear layer so the network emits exactly ``num_classes`` logits.

    Args:
        checkpoint_path (str | os.PathLike):
            Path to a ``.pt`` or ``.pth`` file produced by ``torch.save``.
            The file must contain either a full model state-dict or a dict
            with a ``"state_dict"`` key (common in Lightning checkpoints).
        num_classes (int, optional):
            Number of output classes.  Defaults to 2 (Normal, Pneumonia).
        device (str | torch.device, optional):
            Target device, e.g. ``'cpu'``, ``'cuda'``, or ``'cuda:0'``.
            Defaults to ``'cpu'`` so the function is always safe to call even
            without a GPU present.

    Returns:
        torch.nn.Module:
            The DenseNet121 in eval mode, with weights loaded and residing on
            ``device``.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
        RuntimeError: If the state-dict keys are incompatible with the model.

    Example:
        >>> model = load_model("weights/densenet_pneumonia.pth",
        ...                    num_classes=2, device="cuda")
    """
    # Build with pretrained=False — we supply our own weights.
    # Using pretrained=True here would download ImageNet weights and then
    # immediately overwrite them, wasting time and bandwidth.
    model = models.densenet121(pretrained=False)

    # DenseNet121 stores its final classifier as model.classifier.
    # in_features is 1024 for this architecture (output of the global
    # average-pool after the dense blocks).  We capture it dynamically
    # rather than hard-coding 1024 so the code stays correct if someone
    # swaps to DenseNet169/201 in the future.
    in_features = model.classifier.in_features  # 1024 for DenseNet121
    model.classifier = nn.Linear(in_features, num_classes)

    # map_location ensures the checkpoint loads onto the correct device even
    # if it was originally saved on a different device (e.g. saved on GPU,
    # loaded on CPU).  Without this, PyTorch raises a RuntimeError when a
    # CUDA tensor is loaded on a CPU-only machine.
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support two common checkpoint formats:
    #   1. A bare state-dict  (e.g. torch.save(model.state_dict(), path))
    #   2. A wrapper dict     (e.g. {"state_dict": ..., "epoch": ..., ...})
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    # eval() disables dropout and batch-norm running-stat updates.
    # This is critical for reproducible inference — batch-norm in train mode
    # uses per-batch statistics, which makes predictions non-deterministic.
    model.eval()

    # Move the entire model (weights + buffers) to the target device in one
    # call.  Doing this after load_state_dict avoids an unnecessary CPU->GPU
    # copy of the original random weights.
    model = model.to(device)

    return model


def get_target_layer(model):
    """
    Return the convolutional layer inside DenseNet121 that Grad-CAM should
    hook into.

    Why this layer?
    ---------------
    Grad-CAM works by computing the gradient of the class score with respect
    to the *activations* of a chosen convolutional layer, then weighting those
    activation maps by their gradient-derived importance.

    The final convolutional layer of the last dense block
    (``denseblock4.denselayer16.conv2``) is the ideal target because:

    * It has the highest semantic content — it has "seen" the entire receptive
      field of the network and encodes high-level pathology features.
    * It still has spatial resolution (7x7 for a 224x224 input) so the
      resulting heatmap can be meaningfully upsampled back to the input size.
    * Layers deeper than this are batch-norm / ReLU / pool ops that either
      destroy gradient information or have no spatial extent.

    Args:
        model (torch.nn.Module):
            A DenseNet121 instance, typically returned by :func:`load_model`.

    Returns:
        torch.nn.Module:
            The ``conv2`` layer of ``denselayer16`` inside ``denseblock4``.
            This is the object that Grad-CAM registers its forward/backward
            hooks on.

    Example:
        >>> target = get_target_layer(model)
        >>> print(target)
        Conv2d(...)
    """
    # Accessing nested sub-modules directly (rather than iterating over
    # named_modules) is intentional: it makes the target explicit and raises
    # an AttributeError immediately if the architecture ever changes, rather
    # than silently returning the wrong layer.
    return model.features.denseblock4.denselayer16.conv2


def predict(model, image_tensor, device, class_names=None):
    """
    Run a single forward pass and return human-readable prediction results.

    The function deliberately keeps all tensor operations inside
    ``torch.no_grad()`` so that no computation graph is built.  This halves
    memory usage during inference and is especially important when processing
    large batches of X-rays.

    Note: this function expects a *single* image tensor (or a batch of size 1).
    For batch inference over many images see ``examples/run_batch.py``.

    Args:
        model (torch.nn.Module):
            A loaded, eval-mode DenseNet121, e.g. from :func:`load_model`.
        image_tensor (torch.Tensor):
            A float tensor of shape ``(1, C, H, W)`` or ``(C, H, W)``.
            Values should be normalised to the ImageNet mean/std that the
            model was trained with.  The function will add a batch dimension
            if one is missing.
        device (str | torch.device):
            The device on which ``model`` lives.  ``image_tensor`` will be
            moved here automatically so the caller does not need to track
            device placement manually.
        class_names (list[str], optional):
            Human-readable label for each output logit, in the same order as
            the model's output neurons.  Defaults to
            ``['Normal', 'Pneumonia']``.

    Returns:
        dict: A result dictionary with three keys:

        * ``"predicted_class"`` (str) --
          The class name with the highest softmax probability.
        * ``"confidence"`` (float) --
          The softmax probability of the predicted class, in ``[0, 1]``.
        * ``"probabilities"`` (list[float]) --
          Softmax probabilities for *all* classes, in the same order as
          ``class_names``.

    Example:
        >>> result = predict(model, img_tensor, device="cpu")
        >>> print(result["predicted_class"], f"{result['confidence']:.1%}")
        Pneumonia 94.3%
    """
    if class_names is None:
        class_names = ["Normal", "Pneumonia"]

    # Accept both (C, H, W) and (1, C, H, W) — add the batch dim when absent
    # so downstream code always operates on 4-D tensors.
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)

    # Move the tensor to the same device as the model.  Keeping this inside
    # predict() means callers can pass CPU tensors without worrying about
    # where the model lives — useful in multi-GPU pipelines.
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Forward pass produces raw logits (un-normalised scores).
        logits = model(image_tensor)  # shape: (1, num_classes)

        # Softmax converts logits to a proper probability distribution that
        # sums to 1.  We use dim=1 to operate over the class axis.
        # Note: we do NOT use log_softmax here because we want interpretable
        # probabilities, not log-probabilities.
        probabilities = F.softmax(logits, dim=1)  # shape: (1, num_classes)

    # Move to CPU and convert to a plain Python list so the return value is
    # JSON-serialisable and doesn't hold a reference to the GPU memory.
    prob_list = probabilities.squeeze(0).cpu().tolist()  # list[float]

    # argmax gives the index of the highest probability class.
    predicted_index = int(probabilities.argmax(dim=1).item())

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": prob_list[predicted_index],
        "probabilities": prob_list,
    }
