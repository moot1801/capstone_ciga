from typing import Tuple, Literal
from dataclasses import dataclass


@dataclass
class OptimizationParams:
    position_lr_init: float = 0.00016  #카메라 위치 학습률 초기값
    position_lr_final: float = 0.0000016  # 최종값
    position_lr_delay_mult: float = 0.01  # 지연 곱
    position_lr_max_steps: float = 30_000   # 최대 스텝 수
    feature_lr: float = 0.0025  #피처 학습률
    feature_rest_lr_init: float = 0.0025 / 20.   #피처 나머지 학습률 초기값
    feature_rest_lr_final_factor: float = 0.1  #최종값
    feature_rest_lr_max_steps: int = -1    #최대 스텝 수 (-1: 무한)
    feature_extra_lr_init: float = 1e-3   #피처 추가 학습률 초기값
    feature_extra_lr_final_factor: float = 0.1   #학습률 최종값
    feature_extra_lr_max_steps: int = 30_000   #최대 스텝 수
    opacity_lr: float = 0.05    #불투명도 학습률
    scaling_lr: float = 0.005   #스케일링 학습률
    rotation_lr: float = 0.001   #회전 학습률

    spatial_lr_scale: float = -1  # auto calculate from camera poses if > 0
