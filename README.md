## 🚧 README Update in Progress

This README is currently being finalized. By the end of tonight (5 Feb) or tomorrow, it will include complete instructions for running the experiments and reproducing the results.

# TipsOmaly (ICASSP 2026)

Official PyTorch implementation of [TIPS Over Tricks: Simple Prompts for Effective Zero-shot Anomaly Detection](https://arxiv.org/abs/2602.03594) — a spatially-aware zero-shot anomaly detection pipeline built on the [TIPS](https://arxiv.org/abs/2410.16512) vision-language model, using decoupled prompts and local evidence injection to improve image-level and pixel-level performance.

---

## Table of Contents

- [📖 Introduction](#-introduction)
- [📊 Results](#-results)
<!-- - [🚀 Quickstart](#-quickstart) -->
<!-- - [🔧 Setup](#-setup) -->
<!-- - [🗂️ Datasets](#️-datasets) -->
<!-- - [🛠️ Training](#️-training) -->
<!-- - [🔗 Citation](#-citation) -->
<!-- - [🙏 Acknowledgements](#-acknowledgements) -->
<!-- - [⚖️ License](#️-license) -->

---

## 📖 Introduction
Anomaly detection identifies departures from expected behavior in safety-critical settings. When target-domain normal data are unavailable, zero-shot anomaly detection (ZSAD) leverages vision-language models (VLMs). However, CLIP's coarse image-text alignment limits both localization and detection due to (i) spatial misalignment and (ii) weak sensitivity to fine-grained anomalies; prior work compensates with complex auxiliary modules yet largely overlooks the choice of backbone. We revisit the backbone and use TIPS-a VLM trained with spatially aware objectives. While TIPS alleviates CLIP's issues, it exposes a distributional gap between global and local features. We address this with decoupled prompts-fixed for image-level detection and learnable for pixel-level localization-and by injecting local evidence into the global score. Without CLIP-specific tricks, our TIPS-based pipeline improves image-level performance by 1.1-3.9% and pixel-level by 1.5-6.9% across seven industrial datasets, delivering strong generalization with a lean architecture.

![Architecture](imgs/Models_Architecture_page-0001.jpg)

---

## 📊 Results
Across 14 industrial and medical benchmarks, TipsOmaly consistently outperforms prior CLIP-based zero-shot methods while remaining lightweight. The figure below summarizes its performance across datasets.

![Results](imgs/results-table.png)

We compare pixel-level anomaly maps from Tipsomaly with prior CLIP-based methods (AdaCLIP and AnomalyCLIP) on industrial and medical samples. As shown in the following figure, TipsOmaly more accurately localizes anomalous regions across both domains.

![Qualitative-results](imgs/Qualitative_results_page-0001.jpg)

<!-- ## 🚀 Quickstart
To make an inference on a dataset, make sure to have your dataset prepared in the format mentioned in section [Datasets](#️-datasets) and

## 🔧 Setup
setup

## 🗂️ Datasets
datasets

## 🛠️ Training

## 🔗 Citation

## 🙏 Acknowledgements

## ⚖️ License
li
 -->
