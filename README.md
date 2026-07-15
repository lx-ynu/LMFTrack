# LMFTrack

Official implementation of **“Semantic Language-guided Multi-modal Context-Scale Aware and Sparse Fusion for RGB-T Tracking.”**

LMFTrack integrates category-level semantic prompting, context-scale-aware multi-modal interaction, and sparse cross-modal fusion for RGB-T tracking.

<p align="center">
  <img src="assets/LMFTrack.png" width="100%" alt="Overall framework of LMFTrack">
</p>

## Highlights

- **CSPG** selects a template-aware category-level semantic prompt with CLIP.
- **MCSAII** models global contextual dependencies and multi-scale spatial structures.
- **MMSCF** preserves dominant cross-modal channel correspondences through Top-K sparse cross-attention.

## Installation

LMFTrack was developed with **Ubuntu <kbd>22.04</kbd>**, **Python <kbd>3.9</kbd>**, **PyTorch <kbd>1.10.0</kbd>**, and **CUDA <kbd>11.3</kbd>**. CSPG uses OpenAI CLIP ViT-B/32 with **<kbd>512</kbd>-dimensional** image and text embeddings.

```bash
git clone https://github.com/lx-ynu/LMFTrack.git
cd LMFTrack
conda create -n lmftrack python=3.9 -y
conda activate lmftrack
bash install.sh
```

## Data Preparation

The model is trained on the official LasHeR training split and evaluated on GTOT, RGBT234, LasHeR, and VTUAV. The default relative layout is:

```text
LMFTrack/
└── data/
    ├── GTOT/
    ├── RGBT234/
    ├── LasHeR/
    │   ├── trainingset/
    │   ├── testingset/
    │   ├── trainingsetList.txt
    │   └── testingsetList.txt
    └── VTUAV/
```

Custom paths can be set with `LMFTRACK_DATA_DIR` and `LMFTRACK_OUTPUT_DIR`, or edited in `lib/train/admin/local.py` and `lib/test/evaluation/local.py`.

## Data Augmentation

During training, random translation, scale jittering, horizontal flipping, and brightness jittering are applied. The center jitter factor and scale jitter factor are **<kbd>3</kbd>** and **<kbd>0.25</kbd>**, the horizontal-flip probability is **<kbd>0.5</kbd>**, and the brightness jitter factor is **<kbd>0.2</kbd>**. Identical spatial transformations are applied to the aligned RGB and TIR images.

## Pretrained Models

Place the released SOT-pretrained checkpoint in:

```text
pretrained_models/LMFTrack_SOT_Pretrained.pth.tar
```

The original inherited checkpoint name `TBSITrack_SOT_Pretrained.pth.tar` is also recognized automatically for backward compatibility. Final checkpoints are saved as `LMFTrack_epXXXX.pth.tar`.

## Training

```bash
bash train.sh
```

Equivalent command:

```bash
python tracking/train.py \
    --script lmftrack \
    --config lmftrack_lasher \
    --save_dir ./output/lmftrack_lasher \
    --mode single
```

The released configuration uses **<kbd>15</kbd> epochs**, **AdamW**, a batch size of **<kbd>16</kbd>**, a learning rate of **<kbd>1e-4</kbd>**, and a weight decay of **<kbd>1e-4</kbd>**. The default training seed is **<kbd>42</kbd>** and can be changed with `--seed`.

## Evaluation

```bash
bash test.sh          # LasHeR
bash test_gtot.sh     # GTOT
bash test_rgbt234.sh  # RGBT234
bash test_vtuav.sh    # VTUAV
```

The experiments follow the one-pass evaluation protocol and report PR, NPR, and SR.

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
    <td align="center" width="50%"><img src="assets/LMFTrack_LasHeR_Att_PR.png" width="100%" alt="Attribute-based PR comparison"></td>
    <td align="center" width="50%"><img src="assets/LMFTrack_LasHeR_Att_SR.png" width="100%" alt="Attribute-based SR comparison"></td>
  </tr>
</table>

## Citation

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

Xiang Liu: `liuxiang1@stu.ynu.edu.cn`
