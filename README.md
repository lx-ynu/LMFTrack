# LMFTrack

Official implementation of **“Semantic Language-guided Multi-modal Context-Scale Aware and Sparse Fusion for RGB-T Tracking.”**

LMFTrack is a language-guided RGB-T tracking framework that incorporates category-level semantic priors, context-scale-aware multimodal interaction, and selective sparse cross-modal fusion.

## Overview

RGB-T tracking combines RGB and thermal infrared information to improve tracking robustness under challenging conditions such as low illumination, occlusion, adverse weather, scale variation, and background interference.

LMFTrack contains three principal components:

- **Category Semantic Prompt Generator (CSPG):** selects a category-level semantic prompt according to the RGB template currently maintained by the online tracking pipeline.
- **Multi-modal Context-Scale Aware Interaction Integration (MCSAII):** jointly models non-local contextual dependencies and multi-scale spatial structures.
- **Multi-modal Sparse Cross Fusion (MMSCF):** preserves dominant cross-modal channel correspondences through channel-wise Top-K sparse cross-attention.

## Framework

<p align="center">
  <img src="assets/LMFTrack_framework.png" width="95%">
</p>

> Please place the framework figure at `assets/LMFTrack_framework.png`.

## Main Features

### Category Semantic Prompt Generator

CSPG adopts **CLIP ViT-B/32**, whose image and text embeddings are 512-dimensional. The CLIP image encoder and text encoder remain frozen during training and inference.

A shared lightweight adapter is applied to both image and text embeddings. The adapter operates on the 512-dimensional CLIP representations before the selected textual feature is projected to the ViT token dimension.

The complete COCO-80 vocabulary is used as a fixed set of coarse semantic anchors. For each maintained RGB template, CSPG computes the matching probabilities between the visual representation and all candidate category prompts and selects the best-matching prompt through Top-1 routing.

The default prompt template is:

```text
a photo of {class}
