# ARNet-v2
### Assisted Refinement Network Based on Channel Information Interaction for Camouflaged and Salient Object Detection

**Authors:** 

Camouflaged Object Detection (COD) remains a challenging task in computer vision, aiming to identify and segment objects that are visually integrated with their surroundings. Although existing methods have achieved progress in cross-layer feature fusion, two key issues persist during decoding:
(1) insufficient cross-channel information interaction within same-layer features, which limits feature expressiveness, and
(2) ineffective collaborative modeling between boundary and region information, leading to incomplete regions and inaccurate boundaries.

To address these challenges, we propose the Assisted Refinement Network (ARNet) — a dual-dimensional collaborative decoding framework that enhances both feature interaction and boundary–region consistency.

### Key Contributions

Channel Information Interaction Module (CIIM)
Introduces a bidirectional horizontal–vertical integration mechanism at the channel dimension, enabling feature reorganization and interaction across channels to capture complementary cross-channel information effectively.

Boundary Extraction (BE) and Region Extraction (RE) Modules
Generate boundary priors and object localization maps, respectively, and employ Hybrid Guided Attention (HGA) within CIIM to jointly calibrate decoded features, thereby improving boundary sharpness and spatial localization.

Multi-scale Enhancement (MSE) Module
Enriches contextual feature representations using a three-branch multi-scale convolution strategy, expanding the receptive field and improving global–local feature integration.

### Experimental

Extensive experiments on four COD benchmark datasets demonstrate ARNet’s state-of-the-art performance and strong generalization ability.
We further transfer ARNet to the Salient Object Detection (SOD) task and validate its adaptability across various downstream applications, including polyp segmentation, transparent object detection, and industrial/road defect detection.

---

## 📖 Table of Contents
- [Our Motivation](#-our-motivation)
- [Overview](#-overview)
- [Environment](#-environment)
- [Materials & Evaluation](#%EF%B8%8F-materials--evaluation)
- [Experimental Results](#-experimental-results)

---

## 🎯 Our Motivation

<img width="1720" height="801" alt="image" src="https://github.com/user-attachments/assets/964c73a3-28f0-43bd-89d9-45985a2030d4" />


---

## 📝 Overview







---

## 💻 Environment

-   `python = 3.9`
-   Other packages can be found in `requirements.txt`.

---

## 🛠️ Materials & Evaluation

### Required Materials
+   **Datasets:** You can find the [training and test datasets](https://github.com/DengPingFan/SINet/) here.
+   **SMT Weights:** Download from [Google Drive](https://drive.google.com/file/d/1F8E_Ca6nvusNjp0SqWBVyImWsUSMTSN1/view?usp=sharing).
+   **Pre-trained Model:** Download our weights from [Google Drive](https://drive.google.com/file/d/18xlY7ZtTwPk7MCdYnGMIpj1hafc0GZBq/view?usp=sharing).
+   **Prediction Results:**
    +   [COD Results (Google Drive)](https://drive.google.com/file/d/1mLujes7kTj_6BZrdvfTEGhk4EzN6dttR/view?usp=sharing)
    +   [SOD Results (Google Drive)](https://drive.google.com/file/d/1yM52z75vYh058-c00wG7dfgtd8-2oVpL/view?usp=sharing)
    +   [Polyp Results (Google Drive)](https://drive.google.com/file/d/1jY4_BzgSZzjrtztGtZt9_PEIqJaFfO0L/view?usp=sharing)




### Evaluation Script
We provide a simple script to reproduce the quantitative results.

#### 📂 File Structure
Please organize your files as follows to run the script without any modifications:

```
.
├── eval
├── evaluation
│   └── evaluation.py
├── TestDataset
│   └── NC4K
│       └── GT
│           ├── <mask_1>.png
│           ├── <mask_2>.png
│           └── ...
└── test_maps
    └── ARNet
        └── NC4K
            ├── <pred_1>.png
            ├── <pred_2>.png
            └── ...
```
-   `TestDataset/NC4K/GT/`: Contains the ground truth masks for the NC4K dataset.
-   `test_maps/ARNet/NC4K/`: Contains the predicted maps from our method (ARNet) on the NC4K dataset.

#### 🚀 How to Run
Simply execute the `evaluation.py` script to automatically compute and display all evaluation metrics.

```bash
python evaluation.py
```
Note: You can also modify the mask_root and pred_root variables within the script to evaluate different datasets or methods.

## 📊 Experimental Results

<img width="1091" height="592" alt="image" src="https://github.com/user-attachments/assets/69143375-3412-477d-955a-6547e759431a" />


