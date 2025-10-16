from typing import Literal
from dataclasses import dataclass
import numpy as np


@dataclass
class TCNNEncodingConfig:
    type: Literal["frequency", "hashgrid", "densegrid", "identity", "SphericalHarmonics", "none"] = "hashgrid"  #사용할 인코딩 방식(이중 하나)
    # frequency config
    n_frequencies: int = 4     #frequency encoding일 때 주파수 개수
    # Spherical Harmonics config  사용할 차수(4차 = 0~3)
    degree: int = 4
    # hashgrid config
    n_features_per_level: int = 4  #각 레벨당 피처 수
    log2_hashmap_size: int = 19
    max_resolution: int = 2048
    # both hashgrid and densegrid config
    n_levels: int = 8
    base_resolution: int = 16
    # densegrid config
    per_level_scale: float = 1.405

    def get_encoder_config(self, n_input_channels: int):  
        #입력채널 수를 받아서 선택된 인코딩 방식에 맞는 딕셔너리를 반환
        if self.type == "frequency":   #인코딩 방식 별로 필요한 파라미터 자동 세팅
            return {
                "n_dims_to_encode": n_input_channels,
                "otype": "Frequency",
                "n_frequencies": self.n_frequencies,
            }

        if self.type == "SphericalHarmonics":
            return {
                "n_dims_to_encode": n_input_channels,
                "otype": "SphericalHarmonics",
                "degree": self.degree,
            }

        if self.type == "hashgrid":
            max_res = self.max_resolution
            min_res = self.base_resolution
            num_levels = self.n_levels
            growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
            return {
                "n_dims_to_encode": n_input_channels,
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": self.n_features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": growth_factor,
            }

        if self.type == "densegrid":
            return {
                "n_dims_to_encode": n_input_channels,
                "otype": "DenseGrid",
                "n_levels": self.n_levels,
                "base_resolution": self.base_resolution,
                "per_level_scale": self.per_level_scale,
            }

        return {
            "n_dims_to_encode": n_input_channels,
            "otype": "Identity",
        }
