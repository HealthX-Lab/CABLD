# CABLD: Contrast-Agnostic Brain Landmark Detection with Consistency-Based Regularization

**[Health-X Lab](http://www.healthx-lab.ca/)** | **[IMPACT Lab](https://users.encs.concordia.ca/~impact/)** 

[Soorena Salari](https://soorenasalari.github.io/), [Arash Harirpoush](https://arashharirpoosh.github.io/), [Hassan Rivaz](https://users.encs.concordia.ca/~hrivaz/), [Yiming Xiao](https://yimingxiao.weebly.com/curriculum-vitae.html)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.17845)
[![Overview](https://img.shields.io/badge/Overview-Read-blue.svg)](#overview)
[![BibTeX](https://img.shields.io/badge/BibTeX-Cite-blueviolet.svg)](#citation)
---

## 🧠 Overview

![main figure](https://github.com/HealthX-Lab/CABLD/blob/main/Images/QualitativeResult_H.png)
> **<p align="justify"> Abstract:** *Anatomical landmark detection in medical images is essential for various clinical and research applications, including disease diagnosis and surgical planning. However, manual landmark annotation is time-consuming and requires significant expertise. Existing deep learning (DL) methods often require large amounts of well-annotated data, which are costly to acquire. In this paper, we introduce CABLD, a novel self-supervised DL framework for 3D brain landmark detection in unlabeled scans with varying contrasts by using only a single reference example. To achieve this, we employed an inter-subject landmark consistency loss with an image registration loss while introducing a 3D convolution-based contrast augmentation strategy to promote model generalization to new contrasts. Additionally, we utilize an adaptive mixed loss function to schedule the contributions of different sub-tasks for optimal outcomes. We demonstrate the proposed method with the intricate task of MRI-based 3D brain landmark detection. With comprehensive experiments on four diverse clinical and public datasets, including both T1w and T2w MRI scans at different MRI field strengths, we demonstrate that CABLD outperforms the state-of-the-art methods in terms of mean radial errors (MREs) and success detection rates (SDRs). Our framework provides a robust and accurate solution for anatomical landmark detection, reducing the need for extensively annotated datasets and generalizing well across different imaging contrasts.* </p>


## Method
![main figure](https://github.com/HealthX-Lab/CABLD/blob/main/Images/Workflow.png)
### ✨ Key Features
1) **One-Shot Contrast-Agnostic Landmark Detection:** CABLD detects 3D brain landmarks from unlabeled scans using only a single annotated template, eliminating the need for large labeled datasets.
2) **Consistency-Regularized Multi-Task Learning:** Introduces dual inter-subject and subject-template consistency losses alongside a deformable registration loss to enforce anatomically landmark detection.
3) **3D Random Convolution for Contrast Augmentation:** Pioneers the use of 3D random convolution layers for contrast augmentation, enabling robust performance across unseen MRI contrasts without requiring multi-contrast training data.
4) **Clinically Validated and Robust Performance:** Achieves state-of-the-art accuracy on multiple datasets and shows strong generalization to T2w scans, anatomical misalignments, and downstream disease diagnosis (PD/AD) via landmark-based features.

### :ballot_box_with_check: 3D RC for Contrast Augmentation
To improve generalization across different and unseen MRI contrasts, we use 3D random convolutions for contrast augmentation

<p float="left">
  <img src="https://github.com/HealthX-Lab/CABLD/blob/main/Images/RCModel.png" width="100%" />
</p>

<p float="left">
  <img src="https://github.com/HealthX-Lab/CABLD/blob/main/Images/RCConv_kernel1.png" width="100%" />
</p>

## Results

Results reported below show accuracy for three T1w MRI datasets (HCP, OASIS, and SNSX) and one T2w MRI dataset (HCP), which features an unseen contrast.
### Mean Radial Error (MRE) Comparison Across Datasets (mm)
| **Method**             | **HCP T1w** | **OASIS T1w** | **SNSX T1w** | **HCP T2w** |
|-------------------------|:-------:|:-------:|:-------:|:-------:|
| [3D SIFT](https://ieeexplore.ieee.org/abstract/document/929618)           |  39.44 ± 31.02 | 39.08 ± 29.70 | 41.67 ± 31.84 | 54.90 ± 24.51 |
| [NiftyReg](https://www.sciencedirect.com/science/article/pii/S0169260709002533)            |  4.43 ± 2.42 | 8.23 ± 3.29 | 9.61 ± 4.03 | 4.40 ± 2.41 |
| [ANTs (CC)](https://github.com/ANTsX/ANTs)          |  3.85 ± 2.26 | 4.38 ± 2.64 | 6.36 ± 3.28 | — |
| [ANTs (MI)](https://github.com/ANTsX/ANTs)           |  3.65 ± 2.29 | 4.15 ± 2.65 | 6.06 ± 3.22 | **3.91 ± 2.19** |
| [KeyMorph (64 KPs)](https://www.sciencedirect.com/science/article/pii/S1361841523002220)                   |  8.05 ± 4.51 | 8.20 ± 4.64 | 9.73 ± 5.35 | 6.00 ± 2.64 |
| [KeyMorph (128 KPs)](https://www.sciencedirect.com/science/article/pii/S1361841523002220)                  |  5.77 ± 2.91 | 6.41 ± 3.41 | 8.99 ± 4.16 | 8.66 ± 4.29 |
| [KeyMorph (256 KPs)](https://www.sciencedirect.com/science/article/pii/S1361841523002220)                |  5.37 ± 3.12 | 6.44 ± 3.81 | 8.80 ± 5.22 | 6.41 ± 3.06 |
| [KeyMorph (512 KPs)](https://www.sciencedirect.com/science/article/pii/S1361841523002220)                |  4.67 ± 2.47 | 7.15 ± 3.63 | 5.77 ± 3.27 | 5.54 ± 3.31 |
| [BrainMorph ](https://www.melba-journal.org/papers/2025:010.html)               |  4.11 ± 2.30 | 5.28 ± 3.07 | 13.66 ± 18.21 | 4.24 ± 2.43 |
| [uniGradICON](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_70)               |  4.12 ± 2.53 | 4.63 ± 3.00 | 5.27 ± 3.53 | 13.44 ± 3.88 |
| [MultiGradICON](https://link.springer.com/chapter/10.1007/978-3-031-73480-9_1)               |  4.10 ± 2.56 | 4.62 ± 3.01 | 5.21 ± 3.40 | 4.31 ± 2.70 |
| [Fully Sup. 3D CNN](https://arxiv.org/abs/2411.17845)               |  4.65 ± 2.40 | 4.53 ± 2.81 | 6.64 ± 3.86 | — |
| [**CABLD**](https://arxiv.org/abs/2411.17845)  | **3.27 ± 2.24** | **3.89 ± 2.69** | **5.11 ± 3.19** | 3.99 ± 2.25 |

## 🛠 Requirements

- **Python** >= 3.8  
- **PyTorch** >= 2.0  
- **CUDA** >= 11.8 (optional but recommended)  
- **Dependencies**: `SimpleITK`, `MONAI`, `NumPy`, `SciPy`, `Pandas`

Install via pip:

```bash
pip install torch torchvision SimpleITK monai numpy scipy pandas

```
## Citation
If you find this repository useful, please consider giving a star ⭐ and citation:
```bibtex
@article{salari2025cabldcontrastagnosticbrainlandmark,
        title={CABLD: Contrast-Agnostic Brain Landmark Detection with Consistency-Based Regularization},
        author={Salari, Soorena and Harirpoush, Arash and Rivaz, Hassan and Xiao, Yiming},
        booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},  
        year={2025}  
}
```
## :envelope: Contact

For any questions, feel free to contact the corresponding author: soorena.salari@concordia.ca.

## Acknowledgements

Our code builds upon the [KeyMorph](https://github.com/alanqrwang/keymorph), [BrainMorph](https://github.com/alanqrwang/brainmorph), and [AFIDs](https://github.com/afids/afids-data/tree/main) repositories. We are grateful to the authors for making their code publicly available. If you use our model or code, we kindly request that you also consider citing these foundational works.
