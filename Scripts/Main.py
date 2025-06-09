import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from torch.utils.data import DataLoader

import nibabel as nib
import SimpleITK as sitk
import torchio as tio

from utills import ConvNetCoM, ConvBlock
from functions import RCContrastAugmentationWithNonLinearity, mov_register_RC, landmark_consistency_loss, change_to_IMCord


def calculate_alpha(iteration: int, total: int) -> float:
    """
    Annealing factor for landmark loss weight.
    Maps iteration [0, total] to alpha in [-1,1].
    """
    return 2 / (1 + np.exp(-5 * (iteration / total))) - 1


def load_fixed_image(path: Path, device: torch.device, use_intensity: bool):
    """
    Load and preprocess the fixed/template image.
    """
    img = nib.load(str(path))
    data = img.get_fdata()
    tensor = torch.from_numpy(data).unsqueeze(0)
    if use_intensity:
        transform = tio.transforms.RescaleIntensity((0,1), (0.5, 99.5))
        tensor = transform(tensor)
    tensor = tensor.to(device, dtype=torch.float)
    return tensor, img.affine


def load_target_keypoints(csv_path: Path, inv_affine: np.ndarray, device: torch.device):
    """
    Read CSV of ground-truth landmarks, apply inverse affine and normalize to [-1,1].
    """
    df = pd.read_csv(str(csv_path)).iloc[2:,1:4].astype(float)
    pts = df.values
    pts_world = nib.affines.apply_affine(inv_affine, pts)
    pts_round = np.rint(pts_world).astype(int)
    kp = torch.from_numpy(pts_round).unsqueeze(0).to(device, dtype=torch.float)
    # normalize by image dimensions (z,y,x)
    D, H, W = fixed_tensor.shape[-3:]
    kp[...,0] = 2 * (kp[...,0] / D) - 1
    kp[...,1] = 2 * (kp[...,1] / H) - 1
    kp[...,2] = 2 * (kp[...,2] / W) - 1
    return kp


def validate(network, fixed_tensor, fixed_affine, device, return_weights):
    """
    Evaluate current model on test datasets and return mean residual error.
    """
    network.eval()
    residuals = []
    test_root = Path("/home/so_salar/Desktop/Datasets/LandmarkTests")
    for fld in test_root.iterdir():
        if not fld.is_dir():
            continue
        dataset_errors = []
        for nifti_file in (fld / 'Nifti').glob('*.nii*'):
            # load moving image
            moving_img = nib.load(str(nifti_file))
            moving_data = moving_img.get_fdata()
            # read transform (.xfm)
            xf = fld / 'Transformation' / nifti_file.with_suffix('.xfm').name
            mat = _read_transform(xf)
            # load landmarks
            lm_csv = fld / 'Landmarks' / nifti_file.with_suffix('.csv').name
            subj_kp = pd.read_csv(str(lm_csv))[['x','y','z']].values
            # map to normalized coords
            subj_coords = nib.affines.apply_affine(mat, subj_kp)
            subj_coords = nib.affines.apply_affine(np.linalg.inv(moving_img.affine), subj_coords)
            subj_tensor = torch.from_numpy(moving_data).unsqueeze(0)
            subj_tensor = tio.transforms.RescaleIntensity((0,1),(0.5,99.5))(subj_tensor)
            subj_tensor = subj_tensor.to(device).unsqueeze(0)
            # forward
            with torch.no_grad():
                pts = network(subj_tensor) if not return_weights else network(subj_tensor)[0]
            pred_pts = change_to_IMCord(pts, fixed_tensor)[0].cpu().numpy()
            err = np.linalg.norm(pred_pts - subj_coords, axis=1)
            dataset_errors.append(err.mean())
        residuals.append(np.mean(dataset_errors))
    network.train()
    return float(np.mean(residuals))


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dirs = [
        Path("/home/so_salar/Desktop/Datasets/Mix_IXI_OpenNeuro"),
        Path("/home/so_salar/Desktop/Datasets/7TMRI/Scans"),
    ]
    num_samples = 3
    total_epochs = 2500
    lr = 1e-4

    # Initialize model
    network = ConvNetCoM(dim=3, input_ch=1, out_dim=32, norm_type="instance").to(device)
    

    optimizer = Adam(network.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Prepare data
    train_files = []
    for d in data_dirs:
        train_files += [str(f) for f in d.glob('*.nii*')]
    steps_per_epoch = len(train_files) // num_samples
    total_iters = steps_per_epoch * total_epochs

    # Load fixed image and keypoints
    global fixed_tensor
    fixed_tensor, fixed_affine = load_fixed_image(
        Path('TemplateMNI152NLin2009cSym.nii.gz'), device, use_intensity=True
    )
    target_kp = load_target_keypoints(
        Path('tpl-MNI152NLin2009cSym_res-1_desc-groundtruth_afids.csv'),
        np.linalg.inv(fixed_affine), device
    )

    best_loss = float('inf')
    patience, trig = 500, 0

    # Training loop
    for epoch in range(1, total_epochs + 1):
        epoch_loss = 0.0
        recons_loss_acc, lm_loss_acc = 0.0, 0.0

        for it in range(steps_per_epoch):
            samples = random.sample(train_files, num_samples)
            iteration = (epoch - 1) * steps_per_epoch + it
            alpha = calculate_alpha(iteration, total_iters)
            beta = 1 - alpha

            # Build contrast augmentation
            aug = RCContrastAugmentationWithNonLinearity(4, 1, 0.2).to(device)

            recons_losses, lm_losses = [], []
           
            aligned1, moved_kps1, _ = mov_register_RC(
                    network, samples[0], target_kp, aug,
                    downsample=False, scale=2, GammaAug=True,
                    return_weights=False
                )
            aligned2, moved_kps2, _ = mov_register_RC(
                    network, samples[2], target_kp, aug,
                    downsample=False, scale=2, GammaAug=True,
                    return_weights=False
                )
            
            recons_losses=F.mse_loss(fixed_tensor.detach(), aligned1)+F.mse_loss(fixed_tensor.detach(), aligned2)
            lm_losses=landmark_consistency_loss(moved_kps1, target_kp)+\
                      landmark_consistency_loss(moved_kps2, target_kp)+\
                      landmark_consistency_loss(moved_kps1, moved_kps2)


            loss = beta * recons_losses + alpha * lm_losses
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            recons_loss_acc += recons_losses.item()
            lm_loss_acc += lm_losses.item()

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch:4d}: Total={avg_loss:.4e}, Recon={recons_loss_acc/steps_per_epoch:.4e}, "
              f"Landmark={lm_loss_acc/steps_per_epoch:.4e}, LR={scheduler.get_last_lr()[0]:.2e}")



if __name__ == '__main__':
    main()
