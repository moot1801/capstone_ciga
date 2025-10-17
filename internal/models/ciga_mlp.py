import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer

# input : 위치: x, y, z / 거리(가우시안-카메라): s / 방향 : 카메라 dx, dy, dz
# output : sh 가중치 (sh 최고차항 + 1) * 3 (rgb)

class CigaMLP(nn.Module):
  def __init__(self, in_features, sh_max_degree):
    super(CigaMLP, self).__init__()
    self.linear1 = nn.Linear(in_features, 64, bias=True)
    self.linear2 = nn.Linear(64, 64, bias=True)
    self.linear3 = nn.Linear(64, (sh_max_degree + 1) * 3, bias=True)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # !!!embedding!!!
    z1 = self.linear1(x)
    h1 = self.sigmoid(z1)

    # classifier
    logits = self.linear2(h1)
    probs = self.sigmoid(logits)
    return probs