# ARNet-v2
### Assisted Refinement Network Based on Channel Information Interaction for Camouflaged and Salient Object Detection

**Authors:** Kuan Wang, Meng-ge Lu, Yanjun Qin and Xiaoming Tao

---

## 📖 Table of Contents
- [Our Motivation](#-our-motivation)
- [Overview](#-overview)
- [Environment](#-environment)
- [Materials & Evaluation](#-materials-evaluation)
- [Experimental Results](#-experimental-results)

---

## 🎯 Our Motivation

<img width="1187" height="418" alt="image" src="https://github.com/user-attachments/assets/0c31f894-b2d0-42f9-92bd-fc41daa83018" />


---

## 📝 Overview

Camouflaged Object Detection (COD) stands as a significant challenge in computer vision, dedicated to identifying and segmenting objects visually highly integrated with their backgrounds. While current mainstream methods have made progress in cross-layer feature fusion, they generally neglect leveraging cross-channel information differences within the same-layer features. Addressing this critical issue, we propose an Assisted Refinement Network (ARNet) with channel information interaction, achieving feature optimization through a dual-dimensional collaborative decoding architecture. 

Specifically, our network incorporates three innovations:
1.  We design the **Channel Information Interaction Module (CIIM)**, pioneering a bidirectional horizontal-vertical feature integration mechanism at the channel dimension. This captures cross-channel complementary information through feature splitting reorganization.
2.  To enhance semantic consistency during decoding, we introduce the **Boundary Extraction (BE)** module and **Region Extraction (RE)** module. These generate boundary priors and object localization maps, respectively, and employ Hybrid Guided Attention (HGA) within CIIM to jointly calibrate decoded features for boundary and spatial localization.
3.  We design a **Multi-scale Enhancement (MSE)** module employing a three-branch multi-scale convolution strategy to simultaneously expand the receptive field and enrich contextual feature representations.

Extensive experiments across four COD benchmark datasets validate ARNet's effectiveness and state-of-the-art performance. We further transfer it to the Salient Object Detection (SOD) task and demonstrate our method's exceptional adaptability and versatility across downstream tasks including polyp segmentation, transparent object detection, and industrial/road defect detection.

<img width="1180" height="751" alt="image" src="https://github.com/user-attachments/assets/6c0716e1-1c1d-414d-8394-ac85767918b4" />


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
<img width="1175" alt="image" src="https://github.com/user-attachments/assets/7cf0b2b0-72d9-4c37-afbb-2d33293f8845" />

<img width="1174" alt="image" src="https://github.com/user-attachments/assets/fa41ad1d-ec99-4160-9b80-018a60c68035" />
