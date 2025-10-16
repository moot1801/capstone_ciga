from typing import Tuple
from dataclasses import dataclass
from internal.configs.optimization import OptimizationParams


@dataclass
class ModelParams:
    optimization: OptimizationParams
    sh_degree: int = 3  #계수 0~3  0:1개, 1:4개, 2:9개, 3:16개
    extra_feature_dims: int = 0  #추가 피처 차원
