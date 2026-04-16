# xray-gradcam

Grad-CAM explainability for chest X-ray pneumonia detection using DenseNet121.

---

## What this does

This project applies **Gradient-weighted Class Activation Mapping (Grad-CAM)** to a DenseNet121 chest X-ray classifier to produce visual heatmaps that show *which regions of an X-ray the model focused on* when deciding between Normal and Pneumonia.
Instead of returning a bare confidence score, the pipeline overlays a colour-coded activation map directly on the original image — red regions indicate the anatomical areas that most strongly drove the prediction.
This makes the model's reasoning transparent and auditable, which is a prerequisite for clinical deployment and regulatory review of AI-assisted diagnostic tools.

---

## Why explainability matters in medical AI

The FDA's guidance on **AI/ML-based Software as a Medical Device (SaMD)** explicitly requires that developers demonstrate a model's decision-making process is interpretable and that its outputs can be traced to meaningful clinical features rather than spurious correlations in training data.
A black-box model that outputs "Pneumonia — 94% confident" with no spatial justification is not acceptable in a clinical or regulatory setting: a radiologist has no way to verify the prediction, and a regulator has no way to assess safety.
Grad-CAM is one of the standard post-hoc interpretability techniques recommended in the literature precisely because it produces human-readable spatial explanations without requiring changes to the model architecture or training procedure.
Building explainability into the inference pipeline from the start — rather than retrofitting it after deployment — is both an engineering best practice and a regulatory expectation under the FDA's 2021 Action Plan for AI/ML-based SaMD.

---

## Project structure

```
xray-gradcam/
│
├── src/
│   ├── model.py          # DenseNet121 loader, get_target_layer(), predict()
│   ├── gradcam.py        # GradCAM and GradCAMPlusPlus classes with hook lifecycle
│   └── visualize.py      # preprocess_xray, overlay_heatmap, save_visualization,
│                         # generate_batch_report
│
├── tests/
│   ├── conftest.py       # Shared pytest fixtures (tiny_model, TinyCNN)
│   ├── test_gradcam.py   # Hook behaviour, output shape/range, context manager
│   ├── test_model.py     # predict() contract, get_target_layer() identity checks
│   └── test_visualize.py # Overlay shapes, uint8 ranges, normalisation assertions
│
├── examples/
│   ├── run_single.py     # CLI: run Grad-CAM on one image and save the result
│   └── run_batch.py      # CLI: run Grad-CAM over a directory of images
│
├── data/
│   └── test_images/      # Drop input X-rays here (.gitkeep placeholder)
│
├── outputs/              # Generated PNGs written here (.gitkeep placeholder)
├── requirements.txt      # Pinned dependencies
├── setup.py              # Package install config
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/LSaiko/xray-gradcam.git
cd xray-gradcam
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.0+ are recommended.
CUDA is optional — all scripts auto-detect GPU availability and fall back to CPU.

---

## Quick start

```bash
# Explain the model's top prediction on a single X-ray
python examples/run_single.py \
    --image      data/test_images/patient_01.jpg \
    --checkpoint checkpoints/best_model.pt

# Force explanation of class 1 (Pneumonia) regardless of prediction
python examples/run_single.py \
    --image      data/test_images/patient_01.jpg \
    --checkpoint checkpoints/best_model.pt \
    --class-idx  1

# Use Grad-CAM++ for sharper localisation on a GPU
python examples/run_single.py \
    --image      data/test_images/patient_01.jpg \
    --checkpoint checkpoints/best_model.pt \
    --method     gradcam++ \
    --device     cuda
```

The script prints a one-line summary and writes a three-panel PNG (original | heatmap | overlay) to `outputs/result.png`.

```
[device]  Auto-detected: cpu
[model]   Loading weights from: checkpoints/best_model.pt
[gradcam] Using method: gradcam
Prediction: Pneumonia (94.3%) | Explained class: Pneumonia (idx 1)
Heatmap saved to: /absolute/path/outputs/result.png
```

---

## Running tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

The test suite is **fully offline** — no model checkpoint or real X-ray images are required.

| Test file | What it covers |
|---|---|
| `tests/test_gradcam.py` | Hook registration and cleanup, output shape/range, context manager exception safety, class index contract for both GradCAM and GradCAM++ |
| `tests/test_model.py` | `predict()` return schema and softmax correctness; `get_target_layer()` identity, channel count, and hook compatibility on a real DenseNet121 |
| `tests/test_visualize.py` | `overlay_heatmap()` output shape and uint8 range across edge cases; `preprocess_xray()` tensor shape, channel conversion, and ImageNet normalisation correctness |

---

## Architecture

The pipeline follows the standard Grad-CAM procedure applied to DenseNet121's deepest convolutional layer:

```
Input X-ray (JPEG/PNG)
    |
    v  preprocess_xray()
Resize to 224x224 -> ToTensor -> ImageNet Normalize
    |
    v  model forward pass
DenseNet121 backbone
    |
    +-- [forward hook] --> feature_maps saved  (shape: 1 x C x 7 x 7)
    |
    v  logits -> argmax -> target class score
    |
    v  .backward()
    |
    +-- [backward hook] --> gradients saved    (shape: 1 x C x 7 x 7)
    |
    v  Global average pool over spatial dims
alpha_k = mean(gradients, dim=(H,W))           (shape: 1 x C)
    |
    v  Weighted sum of feature maps
CAM = sum_k(alpha_k * feature_maps_k)          (shape: 7 x 7)
    |
    v  ReLU  (keep only excitatory regions)
    |
    v  Min-max normalise to [0, 1]
    |
    v  overlay_heatmap()
cv2.resize to 224x224 -> COLORMAP_JET -> addWeighted blend
    |
    v  save_visualization()
Three-panel PNG: [ Original ] [ Heatmap ] [ Overlay + Prediction ]
```

**Why `denseblock4.denselayer16.conv2`?**
This is the final convolutional layer before DenseNet121's global average-pool.
It has the highest semantic content in the network — every dense connection from every prior layer feeds into it — while still retaining a 7x7 spatial map that can be upsampled into a meaningful anatomical heatmap.
Hooking earlier layers would give finer spatial resolution but lower semantic relevance; hooking the classifier head gives no spatial information at all.

---

## Connection to Opti-TracT

Opti-TracT (PCT/CA2024/051579) is a patent-pending surgical instrument tracking system that uses computer vision to detect and localise instruments in an operative field in real time.
Both Opti-TracT and xray-gradcam solve the same underlying problem in medical AI: it is not sufficient to know *what* a model detected — a regulatory-compliant clinical system must also be able to show *where* in the image the evidence was found and *why* the model made that call.
Grad-CAM is the explainability layer that would sit directly above any detection or classification backbone in such a pipeline: after the model flags an instrument position or a pathological finding, the CAM heatmap provides the spatial audit trail required by FDA SaMD guidance to demonstrate that the prediction is grounded in clinically relevant image features rather than background artefacts or dataset bias.
In a unified surgical AI platform, this means the same `GradCAM` / `GradCAMPlusPlus` infrastructure developed here could wrap Opti-TracT's detection head to produce frame-by-frame saliency maps — giving surgeons and regulators a transparent, real-time view of what the tracking system is attending to at every moment of a procedure.

---

## Methods available

| | **GradCAM** | **GradCAM++** |
|---|---|---|
| **Paper** | Selvaraju et al., 2017 | Chattopadhay et al., 2018 |
| **Weight formula** | `alpha_k = mean(gradients)` | Second-order correction using gradient curvature |
| **Heatmap style** | Broad region highlight | Sharper, tighter localisation |
| **Best for** | Single dominant lesion; quick sanity-check | Multiple lesions in one image; fine-grained localisation |
| **Speed** | Slightly faster (simpler weight computation) | Marginally slower (extra grad^2 and grad^3 terms) |
| **CLI flag** | `--method gradcam` (default) | `--method gradcam++` |

**Rule of thumb:** start with `gradcam` to confirm the model is looking at the right general region, then switch to `gradcam++` when you need to distinguish between bilateral infiltrates or highlight a specific nodule rather than the entire lung field.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

This project is intended for research and portfolio demonstration.
It is **not** a certified medical device and must not be used for clinical diagnosis without appropriate regulatory approval.
