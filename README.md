# LMFTrack

Official implementation of **“Semantic Language-guided Multi-modal Context-Scale Aware and Sparse Fusion for RGB-T Tracking.”**

LMFTrack is a language-guided RGB-T tracking framework that integrates category-level semantic prompting, context-scale-aware multi-modal interaction, and sparse cross-modal fusion.

<p align="center">
  <img src="assets/LMFTrack.png" width="100%" alt="Overall framework of LMFTrack">
</p>

## Highlights

- **CSPG** selects a template-aware category-level semantic prompt using CLIP.
- **MCSAII** models global contextual dependencies and multi-scale spatial structures.
- **MMSCF** preserves dominant cross-modal channel correspondences through Top-K sparse cross-attention.

## Installation

LMFTrack was developed and evaluated with **Ubuntu <kbd>22.04</kbd>**, **Python <kbd>3.9</kbd>**, **PyTorch <kbd>1.10.0</kbd>**, and **CUDA <kbd>11.3</kbd>**. CSPG uses the official OpenAI CLIP implementation with the **ViT-B/32** backbone and **<kbd>512</kbd>-dimensional** image and text embeddings.

```bash
git clone https://github.com/lx-ynu/LMFTrack.git
cd LMFTrack

conda create -n lmftrack python=3.9 -y
conda activate lmftrack

bash install.sh
```

## Project Paths Setup

Run the following command to configure the workspace, dataset, and output paths:

```bash
python tracking/create_default_local_file.py \
    --workspace_dir . \
    --data_dir ./data \
    --save_dir ./output
```

The paths can also be modified manually in:

```text
lib/train/admin/local.py
lib/test/evaluation/local.py
```

## Data Preparation

LMFTrack is trained on the official LasHeR training split and evaluated on GTOT, RGBT234, LasHeR, and VTUAV.

Place the datasets under `./data`:

```text
LMFTrack/
└── data/
    ├── GTOT/
    ├── RGBT234/
    ├── LasHeR/
    └── VTUAV/
```

## Data Augmentation

During training, random translation, scale jittering, horizontal flipping, and brightness jittering are applied. The center jitter factor and scale jitter factor are set to **<kbd>3</kbd>** and **<kbd>0.25</kbd>**, respectively. The horizontal-flip probability is **<kbd>0.5</kbd>**, and the brightness jitter factor is **<kbd>0.2</kbd>**.

To preserve RGB-T spatial alignment, each RGB-T pair is concatenated into a six-channel input before augmentation, and the same spatial transformations are applied to both modalities.

## Pretrained Models

The ViT backbone is initialized from the pretrained OSTrack checkpoint. Place the pretrained weights under:

```text
pretrained_models/
```

The trained LMFTrack checkpoints and raw tracking results will be provided in this repository.

## Training

```bash
python lib/train/run_training.py \
    --script lmftrack \
    --config baseline
```

LMFTrack is trained for **<kbd>15</kbd> epochs** with **AdamW**, a batch size of **<kbd>16</kbd>**, an initial learning rate of **<kbd>1e-4</kbd>**, and a weight decay of **<kbd>1e-4</kbd>**.

## Evaluation

```bash
python tracking/test.py \
    lmftrack \
    baseline \
    --dataset lasher \
    --threads 4 \
    --num_gpus 1
```

The experiments follow the one-pass evaluation protocol. Precision rate (PR), normalized precision rate (NPR), and success rate (SR) are used for evaluation.

## Results

### Quantitative Comparison

<p align="center">
  <img src="assets/LMFTrack_Results.png" width="95%" alt="Quantitative comparison of LMFTrack">
</p>

### LasHeR Precision and Success Curves

<p align="center">
  <img src="assets/LMFTrack_LasHeR.png" width="100%" alt="Precision and success curves on LasHeR">
</p>

### Attribute-based Evaluation on LasHeR

<table>
  <tr>
    <td align="center" width="50%">
      <img src="assets/LMFTrack_LasHeR_Att_PR.png" width="100%" alt="Attribute-based precision comparison on LasHeR">
      <br>
      <b>Attribute-based PR comparison</b>
    </td>
    <td align="center" width="50%">
      <img src="assets/LMFTrack_LasHeR_Att_SR.png" width="100%" alt="Attribute-based success comparison on LasHeR">
      <br>
      <b>Attribute-based SR comparison</b>
    </td>
  </tr>
</table>

## Citation

If LMFTrack is useful for your research, please consider citing:

```bibtex
@article{liu2026lmftrack,
  title   = {Semantic Language-guided Multi-modal Context-Scale Aware and Sparse Fusion for RGB-T Tracking},
  author  = {Liu, Xiang and Li, Haiyan and Li, Xiangxian and Cao, Jinde and Xie, Shidong and Cai, Jie},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## Acknowledgements

We sincerely thank the authors of [TBSI](https://github.com/RyanHTR/TBSI), [CiteTracker](https://github.com/NorahGreen/CiteTracker), [ViPT](https://github.com/jiawen-zhu/ViPT), [OSTrack](https://github.com/botaoye/OSTrack), and [OpenAI CLIP](https://github.com/openai/CLIP) for making their code and models publicly available.

## Contact

For questions regarding the implementation or experiments, please contact:

- Xiang Liu: `liuxiang1@stu.ynu.edu.cn`
