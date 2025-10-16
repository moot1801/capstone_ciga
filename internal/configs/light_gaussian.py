from typing import List, Literal
from dataclasses import dataclass, field


@dataclass
class LightGaussian:
    prune_steps: List[int] = field(default_factory=lambda: [])  #가지치기 진행할 epoch 리스트
    prune_decay: float = 1.  #가지치기 후 남은 가우시안의 중요도 점수에 곱해지는 값(비율 감쇠 계수)
    prune_percent: float = 0.66  #가지치기 시 제거할 가우시안의 비율
    prune_type: Literal["v_important_score"] = "v_important_score"  #가지치기 방법
    v_pow: float = 0.1  #중요도 점수 계산에 사용되는 지수 파라미턴
