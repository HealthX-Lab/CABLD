# CABLD: Contrast-Agnostic Brain Landmark Detection with Consistency-Based Regularization

This repository contains the implementation of the self-supervised deep learning framework presented in the paper:

**"CABLD: Contrast-Agnostic Brain Landmark Detection with Consistency-Based Regularization"**  
by *Soorena Salari, Arash Harirpoush, Hassan Rivaz, and Yiming Xiao*

---

## 🧠 Overview

CABLD is a data-efficient self-supervised deep learning framework for anatomical landmark detection in 3D brain MRI. It requires only **a single annotated reference scan** and uses landmark consistency across subjects to **generalize across unseen MRI contrasts** (e.g., T1w, T2w).
![System Workflow](https://github.com/HealthX-Lab/CABLD/blob/main/Images/Workflow.png)

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
