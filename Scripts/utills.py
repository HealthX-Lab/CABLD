import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterOfMass3d(nn.Module):
    def __init__(self):
        super(CenterOfMass3d, self).__init__()

    def forward(self, vol):
        """
        x: tensor of shape [n_batch, chs, dimz, dimy, dimx]
        returns: center of mass in normalized coordinates [0,1]x[0,1]x[0,1], shape [n_batch, chs, 2]
        """
        n_batch, chs, dimz, dimy, dimx = vol.shape
        eps = 1e-8
        arangex = (
            torch.linspace(0, 1, dimx).float().view(1, 1, -1).repeat(n_batch, chs, 1)
        )
        arangey = (
            torch.linspace(0, 1, dimy).float().view(1, 1, -1).repeat(n_batch, chs, 1)
        )
        arangez = (
            torch.linspace(0, 1, dimz).float().view(1, 1, -1).repeat(n_batch, chs, 1)
        )

        arangex, arangey, arangez = (
            arangex.to(vol.device),
            arangey.to(vol.device),
            arangez.to(vol.device),
        )

        mx = vol.sum(dim=(2, 3))  # mass along the dimN, shape [n_batch, chs, dimN]
        Mx = mx.sum(dim=-1, keepdim=True) + eps  # total mass along dimN

        my = vol.sum(dim=(2, 4))
        My = my.sum(dim=-1, keepdim=True) + eps

        mz = vol.sum(dim=(3, 4))
        Mz = mz.sum(dim=-1, keepdim=True) + eps

        # center of mass along dimN, shape [n_batch, chs, 1]
        cx = (arangex * mx).sum(dim=-1, keepdim=True) / Mx
        cy = (arangey * my).sum(dim=-1, keepdim=True) / My
        cz = (arangez * mz).sum(dim=-1, keepdim=True) / Mz

        # center of mass, shape [n_batch, chs, 3]
        return torch.cat([cx, cy, cz], dim=-1)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, norm_type, down_sample=True, dim=2
    ):
        super(ConvBlock, self).__init__()
        self.norm_type = norm_type
        self.down_sample = down_sample

        if dim == 2:
            if norm_type == "none":
                self.norm = None
            elif norm_type == "instance":
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm_type == "batch":
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == "group":
                self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
            else:
                raise NotImplementedError()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.down = nn.MaxPool2d(2)

        elif dim == 3:
            if norm_type == "none":
                self.norm = None
            elif norm_type == "instance":
                self.norm = nn.InstanceNorm3d(out_channels)
            elif norm_type == "batch":
                self.norm = nn.BatchNorm3d(out_channels)
            elif norm_type == "group":
                self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
            else:
                raise NotImplementedError()

            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.down = nn.MaxPool3d(2)

        self.activation = nn.ReLU(out_channels)

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        out = self.activation(out)
        if self.down_sample:
            out = self.down(out)
        return out


h_dims = [32, 64, 64, 128, 128, 256, 256, 512]


class ConvNetCoM(nn.Module):
    def __init__(self, dim, input_ch, out_dim, norm_type, return_weights=False):
        super(ConvNetCoM, self).__init__()
        self.dim = dim
        self.return_weights = return_weights
        if self.return_weights:
            self.scales = nn.Parameter(torch.ones(out_dim))
            self.biases = nn.Parameter(torch.zeros(out_dim))

        self.block1 = ConvBlock(input_ch, h_dims[0], 1, norm_type, False, dim)
        self.block2 = ConvBlock(h_dims[0], h_dims[1], 1, norm_type, True, dim)

        self.block3 = ConvBlock(h_dims[1], h_dims[2], 1, norm_type, False, dim)
        self.block4 = ConvBlock(h_dims[2], h_dims[3], 1, norm_type, True, dim)

        self.block5 = ConvBlock(h_dims[3], h_dims[4], 1, norm_type, False, dim)
        self.block6 = ConvBlock(h_dims[4], h_dims[5], 1, norm_type, True, dim)

        self.block7 = ConvBlock(h_dims[5], h_dims[6], 1, norm_type, False, dim)
        self.block8 = ConvBlock(h_dims[6], h_dims[7], 1, norm_type, True, dim)

        self.block9 = ConvBlock(h_dims[7], out_dim, 1, norm_type, False, dim)
        if self.dim == 3:
            self.com = CenterOfMass3d()
        self.relu = nn.ReLU()

    def get_variances(self, heatmap):
        if self.dim == 2:
            return torch.var(heatmap, dim=(2,3))
        else:
            return torch.var(heatmap, dim=(2,3,4))

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        heatmap = self.relu(out)
        points = self.com(heatmap)
        if self.return_weights:
            variances = self.get_variances(heatmap)
            variances = self.scales*variances + self.biases
            return points*2-1, 1/variances
        return points*2-1 #[-1, 1] for F.grid_sample



class TPS:       
  '''See https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/numpy.py'''
  def __init__(self, dim):
      self.dim = dim

  def fit(self, c, lmbda):        
      '''Assumes last dimension of c contains target points.
      
        Set up and solve linear system:
          [K   P] [w] = [v]
          [P^T 0] [a]   [0]
      Args:
        c: control points and target point (bs, T, d+1)
        lmbda: Lambda values per batch (bs)
      '''
      device = c.device
      bs, T = c.shape[0], c.shape[1]
      ctrl, tgt = c[:, :, :self.dim], c[:, :, -1]

      # Build K matrix
      U = TPS.u(TPS.d(ctrl, ctrl))
      I = torch.eye(T).repeat(bs, 1, 1).float().to(device)
      K = U + I*lmbda.view(bs, 1, 1)

      # Build P matrix
      P = torch.ones((bs, T, self.dim+1)).float()
      P[:, :, 1:] = ctrl

      # Build v vector
      v = torch.zeros(bs, T+self.dim+1).float()
      v[:, :T] = tgt

      A = torch.zeros((bs, T+self.dim+1, T+self.dim+1)).float()
      A[:, :T, :T] = K
      A[:, :T, -(self.dim+1):] = P
      A[:, -(self.dim+1):, :T] = P.transpose(1,2)

      theta = torch.linalg.solve(A, v) # p has structure w,a
      return theta
  
  @staticmethod
  def d(a, b):
      '''Compute pair-wise distances between points.
      
      Args:
        a: (bs, num_points, d)
        b: (bs, num_points, d)
      Returns:
        dist: (bs, num_points, num_points)
      '''
      return torch.sqrt(torch.square(a[:, :, None, :] - b[:, None, :, :]).sum(-1) + 1e-6)

  @staticmethod
  def u(r):
      '''Compute radial basis function.'''
      return r**2 * torch.log(r + 1e-6)
  
  def tps_theta_from_points(self, c_src, c_dst, lmbda):
      '''
      Args:
        c_src: (bs, T, dim)
        c_dst: (bs, T, dim)
        lmbda: (bs)
      '''
      device = c_src.device
      
      cx = torch.cat((c_src, c_dst[..., 0:1]), dim=-1)
      cy = torch.cat((c_src, c_dst[..., 1:2]), dim=-1)
      if self.dim == 3:
          cz = torch.cat((c_src, c_dst[..., 2:3]), dim=-1)

      theta_dx = self.fit(cx, lmbda).to(device)
      theta_dy = self.fit(cy, lmbda).to(device)
      if self.dim == 3:
          theta_dz = self.fit(cz, lmbda).to(device)

      if self.dim == 3:
          return torch.stack((theta_dx, theta_dy, theta_dz), -1)
      else:
          return torch.stack((theta_dx, theta_dy), -1)

  def tps(self, theta, ctrl, grid):
      '''Evaluate the thin-plate-spline (TPS) surface at xy locations arranged in a grid.
      The TPS surface is a minimum bend interpolation surface defined by a set of control points.
      The function value for a x,y location is given by
      
        TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_t=0,T theta[t] U(x,y,ctrl[t])
        
      This method computes the TPS value for multiple batches over multiple grid locations for 2 
      surfaces in one go.
      
      Params
      ------
      theta: Nx(T+3)xd tensor, or Nx(T+2)xd tensor
        Batch size N, T+3 model parameters for T control points in dx and dy.
      ctrl: NxTxd tensor
        T control points in normalized image coordinates [0..1]
      grid: NxHxWx(d+1) tensor
        Grid locations to evaluate with homogeneous 1 in first coordinate.
        
      Returns
      -------
      z: NxHxWxd tensor
        Function values at each grid location in dx and dy.
      '''
      
      if len(grid.shape) == 4:
          N, H, W, _ = grid.size()
          diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1)
      else:
          N, D, H, W, _ = grid.size()
          diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1).unsqueeze(1)

      T = ctrl.shape[1]
      
      pair_dist = torch.sqrt((diff**2).sum(-1))
      U = TPS.u(pair_dist)

      w, a = theta[:, :-(self.dim+1), :], theta[:, -(self.dim+1):, :]

      # U is NxHxWxT
      # b contains dot product of each kernel weight and U(r)
      b = torch.bmm(U.view(N, -1, T), w)
      if len(grid.shape) == 4:
          b = b.view(N,H,W,self.dim)
      else:
          b = b.view(N,D,H,W,self.dim)
      
      # b is NxHxWxd
      # z contains dot product of each affine term and polynomial terms.
      z = torch.bmm(grid.view(N,-1,self.dim+1), a)
      if len(grid.shape) == 4:
          z = z.view(N,H,W,self.dim) + b
      else:
          z = z.view(N,D,H,W,self.dim) + b
      return z

  def tps_grid(self, theta, ctrl, size):
      '''Compute a thin-plate-spline grid from parameters for sampling.
      
      Params
      ------
      theta: Nx(T+3)x2 tensor
        Batch size N, T+3 model parameters for T control points in dx and dy.
      ctrl: NxTx2 tensor, or Tx2 tensor
        T control points in normalized image coordinates [0..1]
      size: tuple
        Output grid size as NxCxHxW. C unused. This defines the output image
        size when sampling.
      
      Returns
      -------
      grid : NxHxWx2 tensor
        Grid suitable for sampling in pytorch containing source image
        locations for each output pixel.
      '''    
      device = theta.device
      if len(size) == 4:
          N, _, H, W = size
          grid_shape = (N, H, W, self.dim+1)
      else:
          N, _, D, H, W = size
          grid_shape = (N, D, H, W, self.dim+1)
      grid = self.uniform_grid(grid_shape).to(device)
      
      z = self.tps(theta, ctrl, grid)
      return z 

  def uniform_grid(self, shape):
      '''Uniform grid coordinates.
      
      Params
      ------
      shape : tuple
          NxHxWx3 defining the batch size, height and width dimension of the grid.
          3 is for the number of dimensions (2) plus 1 for the homogeneous coordinate.
      Returns
      -------
      grid: HxWx3 tensor
          Grid coordinates over [-1,1] normalized image range.
          Homogenous coordinate in first coordinate position.
          After that, the second coordinate varies first, then
          the third coordinate varies, then (optionally) the 
          fourth coordinate varies.
      '''

      if self.dim == 2:
          _, H, W, _ = shape
      else:
          _, D, H, W, _ = shape
      grid = torch.zeros(shape)

      grid[..., 0] = 1.
      grid[..., 1] = torch.linspace(-1, 1, W)
      grid[..., 2] = torch.linspace(-1, 1, H).unsqueeze(-1)   
      if grid.shape[-1] == 4:
          grid[..., 3] = torch.linspace(-1, 1, D).unsqueeze(-1).unsqueeze(-1)  
      return grid
  
  def grid_from_points(self, ctl_points, tgt_points, grid_shape, **kwargs):
      lmbda = kwargs['lmbda']

      theta = self.tps_theta_from_points(tgt_points, ctl_points, lmbda)
      grid = self.tps_grid(theta, tgt_points, grid_shape)
      return grid

  def deform_points(self, theta, ctrl, points):
      weights, affine = theta[:, :-(self.dim+1), :], theta[:, -(self.dim+1):, :]
      N, T, _ = ctrl.shape
      U = TPS.u(TPS.d(ctrl, points))

      P = torch.ones((N, points.shape[1], self.dim+1)).float().to(theta.device)
      P[:, :, 1:] = points[:, :, :self.dim]

      # U is NxHxWxT
      b = torch.bmm(U.transpose(1, 2), weights)
      z = torch.bmm(P.view(N,-1,self.dim+1), affine)
      return z + b
  
  def points_from_points(self, ctl_points, tgt_points, points, **kwargs):
      lmbda = kwargs['lmbda']
      theta = self.tps_theta_from_points(ctl_points, tgt_points, lmbda)
      return self.deform_points(theta, ctl_points, points)




    
def change_to_IMCord(Inp_kps,fixed):
    out_kps = (Inp_kps + 1) / 2
    out_kps[:,:,0]=out_kps[:,:,0]*fixed.shape[2]
    out_kps[:,:,1]=out_kps[:,:,1]*fixed.shape[3]
    out_kps[:,:,2]=out_kps[:,:,2]*fixed.shape[4]
    
    return out_kps.int()



# Define the updated RC layer with clamping and absolute value
class RandomConvolutionWithNonLinearity(nn.Module):
    def __init__(self, kernel_size=1, in_channels=1, out_channels=1, min_val=0,
                  max_val=2, clamp_min=0, clamp_max=256,bias=True,min_val_bias=0,
                  max_val_bias=0.01):
        super(RandomConvolutionWithNonLinearity, self).__init__()
        self.max_val=max_val
        self.min_val=min_val
        self.min_val_bias=min_val_bias
        self.max_val_bias=max_val_bias

        self.bias=bias
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=self.bias)
        self.reset_parameters()
        self.clamp_min = clamp_min  # for clamping
        self.clamp_max = clamp_max  # for clamping

    def reset_parameters(self):
        with torch.no_grad():
            self.conv.weight.uniform_(self.min_val, self.max_val)
            self.conv.weight -= (self.max_val - self.min_val) / 2  # zero-centering

            if self.bias:
                self.conv.bias.uniform_(self.min_val_bias, self.max_val_bias)
                self.conv.bias -= (self.max_val_bias - self.min_val_bias) / 2  # zero-centering

    def forward(self, x):
        # Apply convolution
        self.reset_parameters()
        x = self.conv(x)

        return x

# Define the contrast augmentation module with the new non-linear mappings
class RCContrastAugmentationWithNonLinearity(nn.Module):
    def __init__(self, num_layers=4, kernel_size=1,
                  negative_slope=0.2):

        super(RCContrastAugmentationWithNonLinearity, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(RandomConvolutionWithNonLinearity(kernel_size=kernel_size))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))  # LeakyReLU activation
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
def landmark_consistency_loss(registered_landmarks,target_landmarks,
                              return_weights=False,weights=None):
    if return_weights:
            return torch.sum(weights*torch.norm(registered_landmarks - target_landmarks, 
                                         dim=-1))
    else:
        return torch.mean(torch.norm(registered_landmarks - target_landmarks, 
                                 dim=-1))


