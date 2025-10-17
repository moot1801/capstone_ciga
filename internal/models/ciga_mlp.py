import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
      nn.Linear(hidden, sh_max_degree, bias=True)
    )


  def forward(self, x):
    return self.mlp(x)
  
  @classmethod
  def instantiate(cls, **kwargs):
    return cls(**kwargs)
  
  def to_input(self, camera, gaussian_pos):
    """
        print(f"cam_pos:{cam_pos.shape}\n"
              f"{cam_pos}\n"
              f"w2c:{cams.world_to_camera.shape}\n"
              f"{cams.world_to_camera}\n")
                     ->  campos는 카메라 중심 좌표 (배치가 1이라 shape이 3인거임)
                            w2c의 왼쪽 상단 3*3은 회전 R, 3행 0열~2열 : t
                             => 월드좌표계에서 카메라 좌표계로 변환할 때 사용
        am_pos:torch.Size([3])
        tensor([-5.5859e-03, -1.8000e+00, -2.1062e-13], device='cuda:0') 
        w2c:torch.Size([4, 4])
        tensor([[ 2.2204e-16,  0.0000e+00, -1.0000e+00,  0.0000e+00],
                [ 0.0000e+00,  1.0000e+00,  0.0000e+00,  0.0000e+00],
                [ 1.0000e+00,  0.0000e+00,  2.2204e-16,  0.0000e+00],
                [ 2.1062e-13,  1.8000e+00, -5.5859e-03,  1.0000e+00]], device='cuda:0')
        """
    N, 3 = gaussian_pos.shape
  
    cam_pos = camera.camera_center


    return {'cam_pos': ,'dis': ,'dir': }