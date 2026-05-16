"""
gradcam.py — Grad-CAM and Grad-CAM++ implementations for DenseNet121.

Background
----------
Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al. 2017)
produces a coarse heatmap that highlights which spatial regions of an input
image most influenced a CNN's prediction for a given class.

The core insight is that the *gradients* of the class score flowing back into
a convolutional layer act as importance weights for that layer's activation
maps.  Regions with large, positively-weighted activations are the ones the
model "looked at" when making its decision.

Grad-CAM++ (Chattopadhay et al. 2018) refines the weighting scheme so that
individual pixel-level contributions are captured more accurately, which
produces sharper, better-localised heatmaps — especially useful when multiple
instances of the pathology appear in the same image.
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Registers forward and backward hooks on a chosen convolutional layer,
    then uses the captured activations and gradients to produce a spatial
    heatmap showing which regions drove a particular class prediction.

    Typical usage (context manager — hooks are always cleaned up):

        >>> with GradCAM(model, target_layer) as cam:
        ...     heatmap, class_idx = cam.generate(image_tensor)

    Or manually:

        >>> cam = GradCAM(model, target_layer)
        >>> heatmap, class_idx = cam.generate(image_tensor, class_idx=1)
        >>> cam.remove_hooks()
    """

    def __init__(self, model, target_layer):
        """
        Register forward and backward hooks on ``target_layer``.

        We store two pieces of information during the forward/backward pass:

        * **feature_maps**: the raw activation tensor produced by the layer on
          the forward pass.  Shape: ``(1, C, H, W)``.
        * **gradients**: the gradient of the target class score with respect to
          those activations, produced by the backward pass.  Same shape.

        Args:
            model (torch.nn.Module):
                The network to explain.  Must be in eval mode for hooks to
                fire correctly.
            target_layer (torch.nn.Module):
                The convolutional layer to hook.  For DenseNet121 this should
                be the object returned by ``get_target_layer(model)`` from
                ``model.py``.
        """
        self.model = model
        self.target_layer = target_layer

        # Placeholders populated by the hooks below.
        self.feature_maps = None  # forward activations,  (1, C, H, W)
        self.gradients = None     # backward gradients,   (1, C, H, W)

        # --- Forward hook ---
        # PyTorch calls this function immediately after target_layer computes
        # its output during a forward pass.  The signature is fixed by the
        # framework: (module, input, output).
        def _forward_hook(module, input, output):
            # .detach() prevents the saved tensor from being part of the
            # autograd graph.  We only need the *values* of the activations,
            # not their gradient history, so detaching avoids unnecessary
            # memory retention.
            self.feature_maps = output.detach()

        # --- Backward hook ---
        # PyTorch calls this during the backward pass, passing three args:
        #   grad_input  — gradients w.r.t. the layer's *inputs*
        #   grad_output — gradients w.r.t. the layer's *output* (activations)
        #
        # WHY grad_output[0] and not grad_input?
        # ----------------------------------------
        # Grad-CAM needs "how much does each activation map pixel matter for
        # the class score?" — that is exactly dScore/d(output), which lives in
        # grad_output.  grad_input would give dScore/d(input), i.e. how the
        # *preceding* layer's output matters, which is one step further back
        # in the chain rule and not what we want.
        #
        # grad_output is a tuple because a layer can theoretically have
        # multiple outputs (e.g. LSTM cells).  For standard Conv2d there is
        # exactly one output tensor, so we take index [0].
        def _backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # register_forward_hook / register_full_backward_hook both return
        # "handle" objects.  We keep them so remove_hooks() can cleanly
        # deregister without tearing down the whole model.
        self._forward_handle = target_layer.register_forward_hook(_forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(_backward_hook)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def generate(self, image_tensor, class_idx=None):
        """
        Produce a Grad-CAM heatmap for ``image_tensor``.

        Algorithm (Selvaraju et al. 2017, Eq. 1-3):

        1. Forward pass  → logits + feature maps captured by hook.
        2. Backward pass → gradients captured by hook.
        3. Global-average-pool the gradients over spatial dims to get a scalar
           importance weight ``alpha_k`` per channel k.
        4. Weighted sum of feature maps  →  raw CAM.
        5. ReLU  →  keep only regions that *increase* the class score.
        6. Normalise to [0, 1]  →  heatmap ready for overlay.

        Args:
            image_tensor (torch.Tensor):
                Float tensor of shape ``(1, C, H, W)`` or ``(C, H, W)``,
                normalised to ImageNet stats.  If 3-D, a batch dimension is
                added automatically.
            class_idx (int | None, optional):
                Index of the class to explain.  If ``None`` (default), the
                class with the highest logit is used — i.e. explain the
                model's actual prediction.

        Returns:
            tuple(numpy.ndarray, int):
                * ``cam``       — float32 array of shape ``(H_feat, W_feat)``
                  with values in ``[0, 1]``.  Typically 7x7 for a 224px input
                  on DenseNet121.  Upsample to input size before overlaying.
                * ``class_idx`` — the class index that was explained.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # ensure (1, C, H, W)

        # ---- Step 1: Forward pass ----------------------------------------
        # Ensure the input tensor participates in the computation graph.
        # torch.randn() and PIL-loaded tensors have requires_grad=False by
        # default.  When no input to the hooked layer requires gradients,
        # register_full_backward_hook emits a UserWarning about firing "with
        # respect to module outputs since no inputs require gradients."
        # Setting requires_grad_(True) builds the full graph from input →
        # activations → logits, eliminating the warning.
        #
        # We detach() first to sever any prior computation history (e.g. if
        # the caller reuses a tensor across calls) and avoid double-grad
        # issues.  The clone is not needed because we never write in-place.
        image_tensor = image_tensor.detach().requires_grad_(True)

        # The forward hook fires here and populates self.feature_maps.
        # We do NOT use torch.no_grad() because we need gradients to flow back
        # through the network in step 3.
        logits = self.model(image_tensor)  # (1, num_classes)

        # ---- Step 2: Resolve class index ---------------------------------
        if class_idx is None:
            # Use the class the model is most confident about so that the
            # heatmap explains the actual prediction, not an arbitrary class.
            class_idx = int(logits.argmax(dim=1).item())

        # ---- Step 3: Backward pass ---------------------------------------
        # Zero existing gradients first — accumulated gradients from a
        # previous call would corrupt the weights we compute in step 4.
        self.model.zero_grad()

        # We differentiate a *scalar*: the logit for the target class.
        # This propagates dScore_c/d(activation) back to the hooked layer,
        # populating self.gradients via the backward hook.
        logits[0, class_idx].backward()

        # ---- Step 4: Compute channel importance weights ------------------
        # self.gradients shape: (1, C, H_feat, W_feat)
        #
        # WHAT the weights represent mathematically:
        #   alpha_k^c = (1 / Z) * sum_{i,j} dY^c / dA^k_{ij}
        #
        # where Y^c is the class-c logit, A^k is the k-th activation map,
        # and Z = H_feat * W_feat is the number of spatial positions.
        #
        # Intuitively: alpha_k^c is the average gradient across the spatial
        # extent of channel k.  A large positive alpha_k^c means channel k's
        # activations, on average, push the class-c score up — so it's an
        # important channel for class c.
        #
        # Global average pooling over dims (2, 3) collapses (1, C, H, W)
        # to (1, C), giving one scalar weight per channel.
        weights = self.gradients.mean(dim=(2, 3))  # (1, C)

        # ---- Step 5: Weighted sum of feature maps ------------------------
        # self.feature_maps shape: (1, C, H_feat, W_feat)
        # We accumulate: cam = sum_k( alpha_k^c * A^k )
        #
        # Using an explicit for loop (rather than einsum or matmul) keeps the
        # operation readable and avoids a potentially large intermediate tensor
        # from broadcasting (1, C) against (1, C, H, W) all at once.
        cam = torch.zeros(
            self.feature_maps.shape[2:],  # (H_feat, W_feat)
            device=self.feature_maps.device,
        )
        for k, w in enumerate(weights[0]):
            cam += w * self.feature_maps[0, k]  # scalar * (H_feat, W_feat)

        # ---- Step 6: ReLU ------------------------------------------------
        # WHY ReLU?
        # We only care about the features that *increase* the target class
        # score (positive activations with positive weights).  Negative values
        # indicate regions that *suppress* the class — highlighting them would
        # be misleading in a "where did the model look?" visualisation.
        # ReLU zeros those out, keeping only the excitatory regions.
        cam = F.relu(cam)

        # ---- Step 7: Normalise to [0, 1] ---------------------------------
        # epsilon (1e-8) prevents division by zero when the CAM is all-zero
        # (e.g. the target layer wasn't activated at all for this input).
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.cpu().numpy().astype(np.float32), class_idx

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def remove_hooks(self):
        """
        Deregister both hooks from the target layer.

        Always call this (or use the class as a context manager) when you are
        done with Grad-CAM.  Leaving hooks registered causes the model to keep
        allocating tensors for feature_maps and gradients on every forward/
        backward pass, which is both a memory leak and a performance penalty.
        """
        self._forward_handle.remove()
        self._backward_handle.remove()

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        """Enable ``with GradCAM(...) as cam:`` usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Remove hooks when leaving the ``with`` block.

        Called whether the block exits normally or via an exception, so hooks
        are always cleaned up — even if ``generate()`` raises mid-way through.
        Returning False (implicitly) re-raises any exception rather than
        suppressing it.
        """
        self.remove_hooks()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ — an improved weighting scheme for sharper localisation.

    Inherits everything from :class:`GradCAM` (hooks, context manager,
    ``remove_hooks``) and only overrides :meth:`generate` to use the
    second-order gradient weighting from Chattopadhay et al. 2018.

    When to prefer Grad-CAM++ over Grad-CAM
    ----------------------------------------
    * Multiple pathology instances appear in one image (e.g. bilateral
      pneumonia) — Grad-CAM tends to average them together; Grad-CAM++
      localises each separately.
    * Fine-grained localisation matters more than broad region highlight.
    * The prediction confidence is high but the Grad-CAM heatmap looks diffuse.
    """

    def generate(self, image_tensor, class_idx=None):
        """
        Produce a Grad-CAM++ heatmap for ``image_tensor``.

        The algorithm is identical to :meth:`GradCAM.generate` except for how
        the per-channel importance weights ``alpha_k`` are computed (step 4).

        Grad-CAM++ weight formula (Chattopadhay et al. 2018, Eq. 19):

            numerator   = (dY^c / dA^k)^2
                        = gradients^2

            denominator = 2 * (dY^c / dA^k)^2
                        + sum_{i,j}[ A^k_{ij} * (dY^c / dA^k)^3 ]
                        + epsilon

            alpha_k^c   = mean_{i,j}[ numerator / denominator ]

        The denominator adds a second-order (curvature) correction term.
        Dividing by the curvature down-weights channels where the gradient
        is already large in many places (spread-out activations) and
        up-weights channels that have sharp, localised gradient spikes —
        producing tighter, more accurate heatmaps.

        Args:
            image_tensor (torch.Tensor):
                Same spec as :meth:`GradCAM.generate`.
            class_idx (int | None, optional):
                Same spec as :meth:`GradCAM.generate`.

        Returns:
            tuple(numpy.ndarray, int):
                Same spec as :meth:`GradCAM.generate`.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Same requires_grad fix as GradCAM.generate() -- see the detailed
        # comment there.  Both subclasses share identical steps 1-3.
        image_tensor = image_tensor.detach().requires_grad_(True)

        # Steps 1-3 are identical to GradCAM — forward, resolve class,
        # backward.  The hooks populate self.feature_maps and self.gradients
        # exactly as before.
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # ---- Step 4 (Grad-CAM++ variant): second-order weights -----------
        # All tensors below have shape (1, C, H_feat, W_feat).
        gradients = self.gradients      # dY^c / dA^k,  (1, C, H, W)
        feature_maps = self.feature_maps  # A^k,          (1, C, H, W)

        grad_sq = gradients ** 2        # element-wise square
        grad_cu = gradients ** 3        # element-wise cube

        # Numerator: spatial mean of squared gradients — (1, C)
        numerator = grad_sq.mean(dim=(2, 3))

        # Denominator: 2 * spatial-mean(grad^2)
        #            + spatial-sum(A^k * grad^3)
        #            + epsilon
        #
        # The spatial sum of (A^k * grad^3) captures how much the feature map
        # values amplify the third-order gradient signal, which reflects the
        # curvature of the class score surface w.r.t. each activation channel.
        # Adding epsilon (1e-8) guards against division-by-zero in background
        # regions where neither gradients nor activations are present.
        denominator = (
            2.0 * grad_sq.mean(dim=(2, 3))
            + (feature_maps * grad_cu).sum(dim=(2, 3))
            + 1e-8
        )  # (1, C)

        # Element-wise division gives one alpha per channel: (1, C)
        weights = numerator / denominator

        # ---- Steps 5-7: weighted sum, ReLU, normalise --------------------
        # Identical to GradCAM from here on.
        cam = torch.zeros(
            feature_maps.shape[2:],
            device=feature_maps.device,
        )
        for k, w in enumerate(weights[0]):
            cam += w * feature_maps[0, k]

        cam = F.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.cpu().numpy().astype(np.float32), class_idx
