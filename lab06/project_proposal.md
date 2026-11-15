# Individual Project Proposal 

**Student name:** Viktoryia Tsitarenka
**Student ID:** 73500 
**Date:** 15.11.2026

## 1. Problem
**Task:** Classify images of recyclable waste into material categories (for instance, cardboard, glass, metal, paper, plastic, trash) to assist sorting in smart bins / recycling facilities.  
**Why it matters:** Automated pre-sorting can reduce contamination and improve recycling efficiency. This is a clear, scoped computer vision classification problem .

## 2. Dataset
- **Dataset:** TrashNet, ~2,500 labeled images across 6 classes.
- **Split:** 70% train / 15% validation / 15% test.
- **Preprocessing:** resize to 224x224 (or 160x160 ), standard augmentation (flip, crop, color jitter), ImageNet normalization when using pretrained backbones.
- **Feasibility:** Small enough to train on a single GPU/Colab in reasonable time; public and can be well-documented.

## 3. Success metric
- **Primary:** Top-1 test accuracy (target >= 85%).
- **Secondary:** Macro F1-score (to ensure balanced performance across classes, since some categories may be underrepresented).
- **Reporting:** confusion matrix and per-class precision to analyze common misclassifications (e.g., glass vs plastic).

## 4. Deployment constraints
- **Latency:** <= 50 ms per image for on-device inference (batch 1), measured on target-like hardware (laptop CPU or edge-class GPU).
- **Model size:** <= 5 MB on disk (post-optimization) to fit embedded storage and enable fast loading.
- **Runtime:** single-image inference, no cloud dependency required for the classification step.

## 5. Planned optimization
To meet the 5 MB / 50 ms constraints while keeping accuracy high, I plan to use **post-training quantization (INT8)** as the primary technique, with **structured/pruning-aware fine-tuning** as a fallback if needed:
1. **Baseline:** train/fine-tune a compact backbone (e.g., MobileNetV3-Small or EfficientNet-Lite0) to establish accuracy.
2. **Optimization:** apply PyTorch dynamic or static INT8 quantization to reduce weight size (~4x) and improve CPU inference latency.
3. **If constraints not met:** apply magnitude/structured pruning ( 30-50% sparsity) followed by fine-tuning, then quantize.
4. **Validation:** re-measure accuracy, model size, and batch-1 latency after each step and report the accuracy-efficiency trade-off.

## 6. Deliverables
- Training/evaluation scripts with config-driven runs and experiment logs (MLflow).
- Final model artifact meeting size/latency targets.
- Short report with results, profiling measurements, and discussion.
