# CABLD: Contrast-Agnostic Brain Landmark Detection with Consistency-Based Regularization

This repository contains the implementation of the self-supervised deep learning framework presented in the paper:

**"CABLD: Contrast-Agnostic Brain Landmark Detection with Consistency-Based Regularization"**  
by *Soorena Salari, Arash Harirpoush, Hassan Rivaz, and Yiming Xiao*

---

## 🧠 Overview

![main figure](https://github.com/HealthX-Lab/CABLD/blob/main/Images/Workflow.png)
> **<p align="justify"> Abstract:** *Anatomical landmark detection in medical images is essential for various clinical and research applications, including disease diagnosis and surgical planning. However, manual landmark annotation is time-consuming and requires significant expertise. Existing deep learning (DL) methods often require large amounts of well-annotated data, which are costly to acquire. In this paper, we introduce CABLD, a novel self-supervised DL framework for 3D brain landmark detection in unlabeled scans with varying contrasts by using only a single reference example. To achieve this, we employed an inter-subject landmark consistency loss with an image registration loss while introducing a 3D convolution-based contrast augmentation strategy to promote model generalization to new contrasts. Additionally, we utilize an adaptive mixed loss function to schedule the contributions of different sub-tasks for optimal outcomes. We demonstrate the proposed method with the intricate task of MRI-based 3D brain landmark detection. With comprehensive experiments on four diverse clinical and public datasets, including both T1w and T2w MRI scans at different MRI field strengths, we demonstrate that CABLD outperforms the state-of-the-art methods in terms of mean radial errors (MREs) and success detection rates (SDRs). Our framework provides a robust and accurate solution for anatomical landmark detection, reducing the need for extensively annotated datasets and generalizing well across different imaging contrasts..* </p>

CABLD is a data-efficient self-supervised deep learning framework for anatomical landmark detection in 3D brain MRI. It requires only **a single annotated reference scan** and uses landmark consistency across subjects to **generalize across unseen MRI contrasts** (e.g., T1w, T2w).

### ✨ Key Features

- **Single Reference Annotation**: Reduces annotation effort.
- **Consistency-Based Learning**: Enforces anatomically meaningful landmark protocols.
- **3D Random Convolution for Contrast Augmentation**: Promotes generalization across unseen MRI contrasts.
---

## 🛠 Requirements

- **Python** >= 3.8  
- **PyTorch** >= 2.0  
- **CUDA** >= 11.8 (optional but recommended)  
- **Dependencies**: `SimpleITK`, `MONAI`, `NumPy`, `SciPy`, `Pandas`

Install via pip:

```bash
pip install torch torchvision SimpleITK monai numpy scipy pandas
