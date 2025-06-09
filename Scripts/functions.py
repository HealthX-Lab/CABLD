import numpy as np
import nibabel as nib
import random
import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
import nibabel as nib
import pandas as pd
from torch.autograd import Variable
import math
from utills import TPS,ClosedFormAffine,RCContrastAugmentationWithNonLinearity
from scipy.stats import loguniform
from torchio.transforms import (Compose,
                                ToCanonical,
                                RescaleIntensity,
                                CropOrPad,
                                RandomGamma)
import torchio as tio
from torch.cuda.amp import autocast, GradScaler



import matplotlib.pyplot as plt
from skimage.filters import gaussian



def align_img(grid, x):
    return F.grid_sample(
        x, grid=grid, mode="bilinear", padding_mode="border", align_corners=False
    )


def view_cm(x_moved,
            x_aligned,
            x,
            cm_pred,
            cm_target,
            epoch,
            suffix,
            image_idx=0,
            PATH=None,
            show_image=False,
            vmin=0,
            vmax=1,
            titles=['Moving', 'Aligned', 'Target', 'CMs Pred', 'CMs Target']):
    """
    Plot images and keypoints
    
    Arguments
    ---------
    x_moved     : moving image
    x_aligned   : aligned image
    x           : target/fixed image
    cm_pred     : keypoints for the moving image
    cm_target   : keypoints for the target/fixed iamge
    epoch       : current epoch
    suffix      : string for naming the output images
    image_idx   : which image to plot out of the batch
    PATH        : where to save the image
    show_image  : display image
    vmin        : min value of the brain images
    vmax        : max value of the brain images
    titles      : title/name of each plot
    """

    # Cross-section
    s = x.shape[-1] // 2
    _x = x
    x = x[image_idx, 0, :, :, s].data.cpu().numpy()
    x_moved = x_moved[image_idx, 0, :, :, s].data.cpu().numpy()
    x_aligned = x_aligned[image_idx, 0, :, :, s].data.cpu().numpy()
    cm_pred = cm_pred[image_idx, 0, :, :, s].data.cpu().numpy()
    cm_target = cm_target[image_idx, 0, :, :, s].data.cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=5)

    ax[0].set_title(titles[0])
    image = np.flipud(x_moved)
    ax[0].imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].axis('off')

    ax[1].set_title(titles[1])
    image = np.flipud(x_aligned)
    ax[1].imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].axis('off')

    ax[2].set_title(titles[2])
    image = np.flipud(x)
    ax[2].imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].axis('off')

    ax[3].set_title(titles[3])
    image = np.flipud(cm_pred)
    ax[3].imshow(image, cmap='Blues')
    ax[3].axis('off')

    ax[4].set_title(titles[4])
    image = np.flipud(cm_target)
    ax[4].imshow(image, cmap='Blues')
    ax[4].axis('off')

    if PATH is not None:
        fig.set_size_inches(30, 12)
        # fig.savefig(PATH + str(epoch) + suffix)
    elif show_image:
        plt.show()
    
    fig.set_size_inches(20, 12)
    plt.show()
    plt.close('all')




def blur_cm_plot(Cm_plot, sigma):
    """
    Blur the keypoints/center-of-masses for better visualiztion
    
    Arguments
    ---------
    Cm_plot : tensor with the center-of-masses
    sigma   : how much to blur

    Return
    ------
        out : blurred points
    """

    n_batch = Cm_plot.shape[0]
    n_reg = Cm_plot.shape[1]
    out = []
    for n in range(n_batch):
        cm_plot = Cm_plot[n, :, :, :]
        blur_cm_plot = []
        for r in range(n_reg):
            _blur_cm_plot = gaussian(cm_plot[r, :, :, :],
                                     sigma=sigma,
                                     mode='nearest')
            _blur_cm_plot = torch.from_numpy(_blur_cm_plot).float().unsqueeze(0)
            blur_cm_plot += [_blur_cm_plot]

        blur_cm_plot = torch.cat(blur_cm_plot, 0)
        out += [blur_cm_plot.unsqueeze(0)]
    return torch.cat(out, 0)


def get_cm_plot(Y_cm, dim0, dim1, dim2):
    """
    Convert the coordinate of the keypoint/center-of-mass to points in an tensor
    
    Arguments
    ---------
    Y_cm : keypoints coordinates/center-of-masses[n_bath, 3, n_reg]
    dim  : dim of the image

    Return
    ------
        out : tensor it assigns value of 1 where keypoints are located otherwise 0
    """

    n_batch = Y_cm.shape[0]

    out = []
    for n in range(n_batch):
        Y = Y_cm[n, :, :]
        n_reg = Y.shape[1]

        axis2 = torch.linspace(-1, 1, dim2).float()
        axis1 = torch.linspace(-1, 1, dim1).float()
        axis0 = torch.linspace(-1, 1, dim0).float()

        index0 = []
        for i in range(n_reg):
            index0.append(torch.argmin((axis0 - Y[2, i]) ** 2).item())

        index1 = []
        for i in range(n_reg):
            index1.append(torch.argmin((axis1 - Y[1, i]) ** 2).item())

        index2 = []
        for i in range(n_reg):
            index2.append(torch.argmin((axis2 - Y[0, i]) ** 2).item())

        cm_plot = torch.zeros(n_reg, dim0, dim1, dim2)
        for i in range(n_reg):
            cm_plot[i, index0[i], index1[i], index2[i]] = 1

        out += [cm_plot.unsqueeze(0)]

    return torch.cat(out, 0)


import torch


def affine_matrix_func(size, device,s=0.2, o=0.001, a=0.785, z=0.1, cuda=True, random=True):
    """
    Creates a random affine matrix that is used for augmentatiojn
    
    Arguments
    ---------
    size  : size of input .size() method
    s     : scaling interval     [1-s,1+s]
    o     : translation interval [-o,o]
    a     : rotation interval    [-a,a]
    z     : shear interval       [-z,z]
    
    Return
    ------
        out : random affine matrix with paramters drawn from the input parameters 
    """

    n_batch = size[0]
    # device = torch.device('cuda:3') if cuda else torch.device('cpu')

    if random:
        scale = torch.FloatTensor(n_batch, 3).uniform_(1-s, 1+s)
        offset = torch.FloatTensor(n_batch, 3).uniform_(-o, o)
        theta = torch.FloatTensor(n_batch, 3).uniform_(-a, a)
        shear = torch.FloatTensor(n_batch, 6).uniform_(-z, z)
    else:
        scale = torch.FloatTensor(n_batch, 3).fill_(1+s)
        offset = torch.FloatTensor(n_batch, 3).fill_(o)
        theta = torch.FloatTensor(n_batch, 3).fill_(a)
        shear = torch.FloatTensor(n_batch, 6).fill_(z)

    ones = torch.ones(n_batch).float()

    if cuda:
        scale = scale.cuda()
        offset = offset.cuda()
        theta = theta.cuda()
        shear = shear.cuda()
        ones = ones.cuda()

    """Scaling"""
    Ms = torch.zeros([n_batch, 4, 4],
                     device=device)
    Ms[:,0,0] = scale[:,0]
    Ms[:,1,1] = scale[:,1]
    Ms[:,2,2] = scale[:,2]
    Ms[:,3,3] = ones

    """Translation"""
    Mt = torch.zeros([n_batch, 4, 4],
                      device=device)
    Mt[:,0,3] = offset[:,0]
    Mt[:,1,3] = offset[:,1]
    Mt[:,2,3] = offset[:,2]
    Mt[:,0,0] = ones
    Mt[:,1,1] = ones
    Mt[:,2,2] = ones
    Mt[:,3,3] = ones

    """Rotation"""
    dim1_matrix = torch.zeros([n_batch, 4, 4], device=device)
    dim2_matrix = torch.zeros([n_batch, 4, 4], device=device)
    dim3_matrix = torch.zeros([n_batch, 4, 4], device=device)

    dim1_matrix[:,0,0] = ones
    dim1_matrix[:,1,1] = torch.cos(theta[:,0])
    dim1_matrix[:,1,2] = -torch.sin(theta[:,0])
    dim1_matrix[:,2,1] = torch.sin(theta[:,0])
    dim1_matrix[:,2,2] = torch.cos(theta[:,0])
    dim1_matrix[:,3,3] = ones

    dim2_matrix[:,0,0] = torch.cos(theta[:,1])
    dim2_matrix[:,0,2] = torch.sin(theta[:,1])
    dim2_matrix[:,1,1] = ones
    dim2_matrix[:,2,0] = -torch.sin(theta[:,1])
    dim2_matrix[:,2,2] = torch.cos(theta[:,1])
    dim2_matrix[:,3,3] = ones

    dim3_matrix[:,0,0] = torch.cos(theta[:,2])
    dim3_matrix[:,0,1] = -torch.sin(theta[:,2])
    dim3_matrix[:,1,0] = torch.sin(theta[:,2])
    dim3_matrix[:,1,1] = torch.cos(theta[:,2])
    dim3_matrix[:,2,2] = ones
    dim3_matrix[:,3,3] = ones

    """Sheer"""
    Mz = torch.zeros([n_batch, 4, 4],
                      device=device)

    Mz[:,0,1] = shear[:,0]
    Mz[:,0,2] = shear[:,1]
    Mz[:,1,0] = shear[:,2]
    Mz[:,1,2] = shear[:,3]
    Mz[:,2,0] = shear[:,4]
    Mz[:,2,1] = shear[:,5]
    Mz[:,0,0] = ones
    Mz[:,1,1] = ones
    Mz[:,2,2] = ones
    Mz[:,3,3] = ones

    Mr = torch.bmm(dim3_matrix, torch.bmm(dim2_matrix, dim1_matrix))
    M = torch.bmm(Mz,torch.bmm(Ms,torch.bmm(Mt, Mr)))
    return M

def close_form_affine(moving_kp, target_kp):
    """
    Obtain affine matrix to align moving keypoints to target keypoints.
    Affine matrix computed in a close form solution. 
    
    Arguments
    ---------
    moving_kp : keypoints from the moving image [n_batch, 3, n_keypoints]
    target_kp : keypoints from the fixed/target image [n_batch, 3, n_keypoints]

    Return
    ------
        out : affine matrix [n_batch, 3, 4]
    """
    Y_cm = moving_kp
    Y_tg = target_kp
    
    # Initialize 
    one = torch.ones(Y_cm.shape[0], 1, Y_cm.shape[2]).float() #Add a row of ones
    one = one.cuda() if Y_cm.is_cuda else one 
    _Y_cm = torch.cat([Y_cm, one],1)    
    
    out = torch.bmm(_Y_cm, torch.transpose(_Y_cm,-2,-1))
    out = torch.inverse(out)
    out = torch.bmm(torch.transpose(_Y_cm,-2,-1), out)
    out = torch.bmm(Y_tg, out)
    return out


def Resize_3DImage(image,new_size):
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetSize(new_size)
    
    image_resampled = resampler.Execute(image)
    # image_resampled=sitk.GetArrayViewFromImage(image_resampled)
    # image_resampled=image_resampled.transpose(2,1,0)
    
    return image_resampled




# Convenience functions
def random_affine_augment(
    img,
    seg=None,
    points=None,
    max_random_params=(0.2, 0.2, 3.1416, 0.1),
    scale_params=None,
):
    """Randomly augment moving image. Optionally augments corresponding segmentation and keypoints.

    :param img: Moving image to augment (bs, nch, l, w) or (bs, nch, l, w, h)
    :param max_random_params: 4-tuple of floats, max value of each transformation for random augmentation.
    :param scale_params: If set, scales parameters by this value. Use for ramping up degree of augmnetation.
    """
    s, o, a, z = max_random_params
    if scale_params:
        assert scale_params >= 0 and scale_params <= 1
        s *= scale_params
        o *= scale_params
        a *= scale_params
        z *= scale_params
    if len(img.shape) == 4:
        scale = torch.FloatTensor(1, 2).uniform_(1 - s, 1 + s)
        offset = torch.FloatTensor(1, 2).uniform_(-o, o)
        theta = torch.FloatTensor(1, 1).uniform_(-a, a)
        shear = torch.FloatTensor(1, 2).uniform_(-z, z)
        augmenter = AffineDeformation2d(device=img.device)
    else:
        scale = torch.FloatTensor(1, 3).uniform_(1 - s, 1 + s)
        offset = torch.FloatTensor(1, 3).uniform_(-o, o)
        theta = torch.FloatTensor(1, 3).uniform_(-a, a)
        shear = torch.FloatTensor(1, 6).uniform_(-z, z)
        augmenter = AffineDeformation3d(device=img.device)

    params = (scale, offset, theta, shear)

    img = augmenter(img, params=params, interp_mode="bilinear")
    res = (img,)
    if seg is not None:
        seg = augmenter(seg, params=params, interp_mode="nearest")
        res += (seg,)
    if points is not None:
        points = augmenter.deform_points(points, params)
        res += (points,)
    return res







def flip_compute_consistency_loss(original_landmarks, flipped_landmarks, axis='x'):
    """
    Compute consistency loss between original and flipped landmarks.
    Args:
    - original_landmarks: Tensor of shape (N, 3), where N is the number of landmarks.
    - flipped_landmarks: Tensor of shape (N, 3), where N is the number of landmarks.
    - axis: Axis along which the image was flipped ('x', 'y', or 'z').
    """
    mirrored_landmarks = flipped_landmarks.clone()
    if axis == 'x':
        mirrored_landmarks[:, :, 0] = -mirrored_landmarks[:, :, 0]  # Mirror the x-coordinate
    elif axis == 'y':
        mirrored_landmarks[:, :, 1] = -mirrored_landmarks[:, :, 1]  # Mirror the y-coordinate
    elif axis == 'z':
        mirrored_landmarks[:, :, 2] = -mirrored_landmarks[:, :, 2]  # Mirror the z-coordinate
    else:
        raise ValueError("Unsupported flip axis: choose from 'x', 'y', or 'z'")
    
    consistency_loss = F.mse_loss(original_landmarks, mirrored_landmarks)
    return consistency_loss



def mov_register(model,moving,
                 fixed_kp,epoch,
                 ResIntens=True,
                 downsample=True,
                 scale=2,
                 GammaAug=False,
                 return_weights=True,
                 FLAug=True):

    if downsample:
        moving=F.interpolate(moving, 
                             scale_factor=1/scale, mode='trilinear')

    """Augment"""
    # Augment
    Ma = affine_matrix_func(moving.size(),
                            s=np.clip(0.2 * epoch / 1, None, 0.2),
                            o=np.clip(0.2 * epoch / 1, None, 0.2),
                            a=np.clip(3.1416 * epoch / 1, None, 3.1416),
                            z=np.clip(0.1 * epoch / 1, None, 0.1)).cuda()

    grid = F.affine_grid(torch.inverse(Ma)[:, :3, :],
                         moving.size(),
                         align_corners=False)

    moving = F.grid_sample(moving,
                           grid=grid,
                           mode='bilinear',
                           padding_mode='border',
                           align_corners=False)
        
 
       
    if return_weights:
        with torch.autocast(device_type="cuda"):
             moving_kp,mov_inv_var=model(moving)
            
    else:
        with torch.autocast(device_type="cuda"):
             moving_kp=model(moving)
       


    # """Close Form Affine Matrix"""
    # affine_matrix = close_form_affine(torch.transpose(moving_kp, 2, 1), 
    #                                   torch.transpose(fixed_kp, 2, 1))
    # inv_matrix = torch.zeros(moving.size(0), 4, 4).cuda() if moving_kp.is_cuda else torch.zeros(moving.size(0), 4, 4)
    # inv_matrix[:, :3, :4] = affine_matrix
    # inv_matrix[:, 3, 3] = 1
    # inv_matrix = torch.inverse(inv_matrix)[:, :3, :]
    
    keypoint_aligner = TPS(3)
    a, b = 1e-6, 10
    tps_lmbda = torch.tensor(loguniform.rvs(a, 
                                            b, size=len(moving))).cuda()
    grid = keypoint_aligner.grid_from_points(
            moving_kp, fixed_kp, moving.shape, lmbda=tps_lmbda
        )
    moved_kps = keypoint_aligner.points_from_points(
                moving_kp, fixed_kp, moving_kp, lmbda=tps_lmbda
            )
    
    
    # inv_grid = keypoint_aligner.grid_from_points(
    #         fixed_kp, moving_kp, moving.shape, lmbda=tps_lmbda
    #     )
    # moved_target_kps = keypoint_aligner.points_from_points(
    #             fixed_kp, moving_kp, fixed_kp, lmbda=tps_lmbda
    #         )
    
    
    """Align Image"""
    # grid = F.affine_grid(inv_matrix,
    #                      moving.size(),
    #                      align_corners=False)

    aligned=F.grid_sample(moving,
                          grid=grid,
                          mode='bilinear',
                          padding_mode='border',
                          align_corners=False)
    
    if FLAug:
       flip_transform = tio.RandomFlip(axes='LR', flip_probability=1.0)
       flipped_moving = flip_transform(moving[0].cpu().detach().numpy())
       flipped_moving=  torch.from_numpy(flipped_moving).cuda().float().unsqueeze(0)
    
    if return_weights:
        if FLAug: 
            with torch.autocast(device_type="cuda"):
                 flipped_moving_kp,_=model(flipped_moving)
            
    else:
        if FLAug: 
            with torch.autocast(device_type="cuda"):
                 flipped_moving_kp=model(flipped_moving)



    
    if return_weights:
        if FLAug:
            return aligned,moved_kps,moving_kp,mov_inv_var,flipped_moving_kp
        else:
            return aligned,moved_kps,moving_kp,mov_inv_var
    else:
        if FLAug:
            return aligned,moved_kps,moving_kp,flipped_moving_kp
        else:
            return aligned,moved_kps,moving_kp
    


def mov_register_RC(model,
                    patient_dir,
                    fixed_kp,
                    rc_augmentation,
                    transforms = Compose([RescaleIntensity((0,1),(0.5, 99.5))]),
                    epoch=1,
                    downsample=False,
                    scale=2,
                    GammaAug=False,
                    return_weights=False,
                    a=1e-6,
                    b=10,
                    keypoint_aligner = TPS(3)
                    ):
    
    moving=nib.load(patient_dir)

    # moving_affine=moving.affine
    moving=moving.get_fdata()
    moving=transforms(torch.from_numpy(moving).unsqueeze(0))
    moving=moving.cuda().float().unsqueeze(0)
    if 'IXI' in patient_dir:moving=torch.transpose(moving,2,4)


  
        
    """Augment"""
    # Augment
    
    Ma = affine_matrix_func(moving.size(),
                            moving.device,
                            s=np.clip(0.2 * epoch / 1, None, 0.2),
                            o=np.clip(0.2 * epoch / 1, None, 0.2),
                            a=np.clip(3.1416 * epoch / 1, None, 3.1416),
                            z=np.clip(0.1 * epoch / 1, None, 0.1)).cuda()

    grid = F.affine_grid(torch.inverse(Ma)[:, :3, :],
                         moving.size(),
                         align_corners=False)


    moving = F.grid_sample(moving,
                           grid=grid,
                           mode='bilinear',
                           padding_mode='border',
                           align_corners=False)
    del grid
 
    augmented_moving=rc_augmentation(moving).cuda()
    augmented_moving=transforms(augmented_moving[0].detach().cpu().numpy())
    augmented_moving=torch.from_numpy(augmented_moving).unsqueeze(0).cuda()
    
    """Mask"""
    #Applying T1 Mask to the Augmented Scan
    mask=(moving>0.0000).detach()
    augmented_moving=augmented_moving*mask
    augmented_moving=augmented_moving.cuda()
    
   
    """Landmark Prediction"""
    with torch.autocast(device_type="cuda"):
          moving_kp=model(augmented_moving)
          # augmented_moving_kp=model(augmented_moving)
          # moving_kp=model(augmented_moving)
          del augmented_moving
          
           
    tps_lmbda = torch.tensor(loguniform.rvs(a, 
                                            b, size=len(moving))).cuda()
    
    grid = keypoint_aligner.grid_from_points(
            moving_kp, fixed_kp, moving.shape, lmbda=tps_lmbda
        )
    
    moved_kps = keypoint_aligner.points_from_points(
                moving_kp, fixed_kp, moving_kp, lmbda=tps_lmbda
            )
   
    
    """Align Image"""
    aligned=F.grid_sample(moving,
                          grid=grid.cuda(),
                          mode='bilinear',
                          padding_mode='border',
                          align_corners=False)
                          
    del grid, tps_lmbda, mask, Ma, moving,moving_kp
    
    return aligned,moved_kps
      
    
