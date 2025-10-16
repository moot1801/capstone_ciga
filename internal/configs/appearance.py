from dataclasses import dataclass
#Neural rendering 모델의 Appearance 파라미터 및 최적화 하이퍼파라미터를 dataclass로 구조화한 것
#각 클래스는 외관 표현용 네트워크 구조 및 옵티마이저 세팅을 구현.

@dataclass
class AppearanceModelOptimizationParams:
    lr: float = 1e-3  #학습률
    eps: float = 1e-15  #수치적 안정성
    gamma: float = 1    #학습률 감쇠 계수(스케일링, 정규화)
    max_steps: int = 30_000  #최대 학습 스텝 수


@dataclass
class AppearanceModelParams:        
    optimization: AppearanceModelOptimizationParams     #최적화 파라미터 객체

    n_grayscale_factors: int = 3  #회색조 요인 수
    n_gammas: int = 3  #감마 보정 수
    n_neurons: int = 32   #신경망 뉴런 수       => 네트워크 깊이/폭
    n_hidden_layers: int = 2  #신경망 은닉층 수  =>이것도
    n_frequencies: int = 4   #주파수 수
    grayscale_factors_activation: str = "Sigmoid"  #회색조 요인 활성화 함수
    gamma_activation: str = "Softplus"  #감마 보정 활성화 함수

@dataclass
class SwagAppearanceModelParams:
    optimization: AppearanceModelOptimizationParams   

    n_appearance_count: int=6000    #appearance 벡터 수
    n_appearance_dims: int = 24   #apparance 벡터 차원 수
    n_input_dims: int = 30  #입력 차원
    n_neurons: int = 64  
    n_hidden_layers: int = 3  
    color_activation: str = "Sigmoid"  #색상 활성화 함수

@dataclass
class VastAppearanceModelParams:
    optimization: AppearanceModelOptimizationParams

    n_appearance_count: int=6000
    n_appearance_dims: int = 64
    n_rgb_dims: int = 3    #rgb 차원 수
    std: float = 1e-4   #표준편차