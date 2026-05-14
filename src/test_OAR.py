import os
import open3d as o3d
import numpy as np
import sys
sys.path.append("../")
import torch
from torch import nn
import pytorch3d
from pytorch3d.io import load_ply
from utils2.normalize_pointcloud import normalize_ply,normalize_ply_file
from utils2.LLR import barycenter_weights, barycenter_kneighbors_graph, local_linear_reconstruction
from utils2.loss_functions import correntropy_chamfer_distance
from model.model import Siren

torch.cuda.set_device(0)
DEVICE = 'cuda'






# deformation learning
def deform_point_cloud(model, xsrc=None, xtrg=None,
                 xsrc_corr=None, xtrg_corr=None,
                 n_samples=10000, n_steps=200, sigma2=1.0, init_lr=1.0e-4,
                 LLR_weight=1.0e2, MCC_chamfer_weight=1.0e4,
                 CORR_weight=1.0e4, CORR_epsilon=0.05,
                 LLR_n_neighbors=30, eval_every_nth_step=100, point_num=None):
  """
  Deform a point cloud using a neural network model.

  Parameters
  ----------
  model : torch.nn.Module
      The neural network model to use for deformation.
  xsrc : numpy.ndarray
      The source point cloud to deform.
  xtrg : numpy.ndarray, optional
      The target point cloud to match (used in MCC distance loss).
  xsrc_corr : torch.Tensor, optional
      User-picked correspondence points on the source cloud, shape (N_corr, 3).
  xtrg_corr : torch.Tensor, optional
      User-picked correspondence points on the target cloud, shape (N_corr, 3).
  n_samples : int, optional
      The number of points to sample for MCC distance loss (default is 10**4).
  n_steps : int, optional
      The number of optimization steps (default is 200).
  init_lr : float, optional
      The initial learning rate for the optimizer (default is 1.0e-4).
  LRR_weight : float, optional
      The weight for LRR loss (default is 1.0e2).
  MCC_chamfer_weight : float, optional
      The weight for chamfer distance loss (default is 1.0e4).
  CORR_weight : float, optional
      The weight for the user-picked correspondence loss (default is 1.0e4).
      Applied to the mean hinge loss over correspondence pairs, so the
      magnitude is independent of N_corr.
  CORR_epsilon : float, optional
      The forgiveness radius in the normalized frame: pairs whose deformed
      source / target L2 distance is below epsilon incur no penalty.
  LLR_n_neighbors: int, optional
      The number of neighbors to use for LRR loss (default is 30).
  eval_every_nth_step : int, optional
      The number of steps between evaluations (default is 100).
  point_num: int, optional
      The minimal number of the two input point clouds
  """

  model = model.train()
  optm = torch.optim.Adam(model.parameters(), lr=init_lr)# optimizer
  schedm = torch.optim.lr_scheduler.ReduceLROnPlateau(optm, verbose=True, patience=1)# lr


  MCC_chamfer_loss_total = 0
  LLR_loss_total = 0
  CORR_loss_total = 0
  total_loss = 0
  n_r = 0

  # Downsampling
  n_samples=5000
  if n_samples>point_num:
      n_samples=point_num

  # Number of user-picked correspondence pairs (coefficient scales as 1/N_corr)
  has_corr = xsrc_corr is not None and xtrg_corr is not None and xsrc_corr.shape[0] > 0
  n_corr = xsrc_corr.shape[0] if has_corr else 0


  for i in range(0, n_steps):
    xbatch_src=xsrc[np.random.choice(len(xsrc), n_samples, replace=False)]
    xbatch_trg=xtrg[np.random.choice(len(xtrg), n_samples, replace=False)]
    xbatch_deformed = xbatch_src + model(xbatch_src)

    loss = 0

    # LLR loss
    LLR_loss = LLR_weight*local_linear_reconstruction(xbatch_src, xbatch_deformed, n_neighbors=LLR_n_neighbors)
    loss += LLR_loss
    LLR_loss_total += float(LLR_loss)


    # MCC
    MCC_loss=correntropy_chamfer_distance(xbatch_deformed.unsqueeze(0),xbatch_trg.unsqueeze(0),sigma2=sigma2)
    MCC_chamfer_loss = MCC_chamfer_weight*MCC_loss
    loss += MCC_chamfer_loss
    MCC_chamfer_loss_total += float(MCC_chamfer_loss)


    # User-picked correspondence loss: mean hinge on the L2 distance
    # with a tolerance epsilon, weighted by CORR_weight.
    if has_corr:
      xcorr_deformed = xsrc_corr + model(xsrc_corr)
      corr_dist = torch.norm(xcorr_deformed - xtrg_corr, dim=-1)
      CORR_loss = CORR_weight * torch.clamp(corr_dist - CORR_epsilon, min=0).mean()
      loss += CORR_loss
      CORR_loss_total += float(CORR_loss)


    total_loss += float(loss)
    n_r += 1

    optm.zero_grad()
    loss.backward()
    optm.step()

    # Evaluate the training results
    if i % eval_every_nth_step == 0:

      LLR_loss_total /= n_r
      MCC_chamfer_loss_total /= n_r
      CORR_loss_total /= n_r
      total_loss /= n_r

      schedm.step(float(total_loss))



      LLR_loss_total = 0
      MCC_chamfer_loss_total = 0
      CORR_loss_total = 0
      total_loss = 0
      n_r = 0

  LLR_loss_total /= n_r
  MCC_chamfer_loss_total /= n_r
  CORR_loss_total /= n_r
  total_loss /= n_r




def MCC_registration(xsrc=None, xtrg=None,
                     xsrc_corr=None, xtrg_corr=None,
                     target_normal_scale=None,target_normal_center=None,
                     n_steps=200,
                     sigma2=1.0,
                     LLR_n_neighbors=30,
                     LLR_WEIGHT=1.0e2,
                     MCC_chamfer_WEIGHT=1.0e4,
                     CORR_WEIGHT=1.0e4,
                     CORR_EPSILON=0.05,
                     out_path=None,
                     point_num=None):


#  define the deformation model
    model = Siren(in_features=3,
                    hidden_features=128,
                    hidden_layers=3,
                    out_features=3, outermost_linear=True,
                    first_omega_0=30, hidden_omega_0=30.).to(DEVICE).train()

    deform_point_cloud(model,
            xsrc=xsrc, xtrg=xtrg,
            xsrc_corr=xsrc_corr, xtrg_corr=xtrg_corr,
            init_lr=1.0e-4,
            n_steps=n_steps,
            sigma2=sigma2,
            LLR_n_neighbors=LLR_n_neighbors,
            LLR_weight=LLR_WEIGHT,
            MCC_chamfer_weight=MCC_chamfer_WEIGHT,
            CORR_weight=CORR_WEIGHT,
            CORR_epsilon=CORR_EPSILON,
            point_num=point_num)
    
    
    model.eval()
    vpred = xsrc + model(xsrc).detach().clone()

    vpred_save=vpred.cpu().numpy()

    vpred_save_denormalize=target_normal_scale*vpred_save+target_normal_center


    pcd_deformed=o3d.geometry.PointCloud()
    pcd_deformed.points=o3d.utility.Vector3dVector(vpred_save_denormalize)

    o3d.io.write_point_cloud(out_path, pcd_deformed)
    print(f"deformed -> {out_path}")



if __name__=='__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_data = os.path.join(BASE_DIR, "data/source/source.ply")
    target_data = os.path.join(BASE_DIR, "data/target/target.ply")
    correspondence_data = os.path.join(BASE_DIR, "data/images/user_correspondence.npy")
    deformed_dir = os.path.join(BASE_DIR, "data/save_deformed/")
    deformed_path = os.path.join(deformed_dir, "deformed.ply")

    os.makedirs(deformed_dir, exist_ok=True)

    print("source:", source_data)
    print("target:", target_data)
    print("correspondence:", correspondence_data)

    # Normalize the input point clouds
    src_normalized_ply, src_normal_center, src_normal_scale = normalize_ply_file(source_data)
    tgt_normalized_ply, tgt_normal_center, tgt_normal_scale = normalize_ply_file(target_data)

    src_points = np.asarray(src_normalized_ply.points, dtype=np.float32)
    tgt_points = np.asarray(tgt_normalized_ply.points, dtype=np.float32)

    src_points = torch.from_numpy(src_points).to(DEVICE)
    tgt_points = torch.from_numpy(tgt_points).to(DEVICE)

    point_num = min(src_points.shape[0], tgt_points.shape[0])

    # Correspondence pairs are (N_corr, 2, 3) world-frame 3D points produced by
    # build_segmented_pointclouds.py from the user's SAM2 foreground clicks
    # ([:, 0] = source, [:, 1] = target, paired by click order). Apply the same
    # per-cloud normalization that produced src_points/tgt_points so the anchors
    # live in the loss frame.
    corr_pairs_3d = np.load(correspondence_data).astype(np.float64)
    src_corr_norm = ((corr_pairs_3d[:, 0, :] - src_normal_center) / src_normal_scale).astype(np.float32)
    tgt_corr_norm = ((corr_pairs_3d[:, 1, :] - tgt_normal_center) / tgt_normal_scale).astype(np.float32)
    xsrc_corr = torch.from_numpy(src_corr_norm).to(DEVICE)
    xtrg_corr = torch.from_numpy(tgt_corr_norm).to(DEVICE)
    print(f"correspondence pairs: {corr_pairs_3d.shape[0]}")

    MCC_registration(xsrc=src_points, xtrg=tgt_points,
                xsrc_corr=xsrc_corr, xtrg_corr=xtrg_corr,
                target_normal_scale=tgt_normal_scale, target_normal_center=tgt_normal_center,
                n_steps=200,
                sigma2=1.0,
                LLR_n_neighbors=30,
                LLR_WEIGHT=1.0e2,
                MCC_chamfer_WEIGHT=1.0e4,
                CORR_WEIGHT=1.0e4,
                CORR_EPSILON=0.05,
                out_path=deformed_path,
                point_num=point_num)

    print("**************************")
