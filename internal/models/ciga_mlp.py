import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer

# input : 위치: x, y, z / 거리(가우시안-카메라): s / 방향 : 카메라 dx, dy, dz
# output : sh 가중치 (sh 최고차항 + 1) * 3 (rgb)


class CigaMLP(nn.Module):
  def __init__(
      self, 
      in_features: int, 
      sh_max_degree: int,
      hidden:int=64
  ):
    super(CigaMLP, self).__init__()
    self.mlp = nn.Sequential(
      nn.Linear(in_features, hidden, bias=True),
      nn.ReLU(),
      nn.Linear(hidden, hidden, bias=True),
      nn.ReLU(),
      nn.Linear(hidden, sh_max_degree+1, bias=True)
    )


  def forward(self, x):
    return self.mlp(x)
  
  @classmethod
  def instantiate(cls, **kwargs):
    return cls(**kwargs)
  
  def to_input(self, camera, gaussian_pos):
    N = (gaussian_pos.shape)[0]
    # 카메라 위치
    cam_pos = camera.camera_center
    cam_pos = cam_pos.unsqueeze(0).expand(N, -1)
    # 카메라-가우시안 거리
    dis = (gaussian_pos - cam_pos).norm(dim=-1, keepdim=True)

    # 카메라-가우시안 방향
    R = camera.world_to_camera[:3, :3]
    dir = torch.tensor([0.0, 0.0, 1.0], device=gaussian_pos.device, dtype=gaussian_pos.dtype) @ R.t()
    dir = F.normalize(dir, dim=0)
    dir = F.normalize(gaussian_pos - cam_pos, dim=1)
    return {'cam_pos': cam_pos, 'dis': dis, 'dir': dir}