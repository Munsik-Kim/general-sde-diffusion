# General SDE Diffusion Framework

**Score-Based Generative Modeling through Stochastic Differential Equations (SDEs)** 논문의 PyTorch 구현체입니다.

이 프로젝트는 **DDPM (VP-SDE)** 과 **Score Matching (VE-SDE)** 을 하나의 코드베이스에서 실험할 수 있도록 설계된 통합 프레임워크입니다. 코드는 교육적 목적을 위해 간결하게 작성되었으며, 확장하기 쉽게 구조화되었습니다.

## ✨ 주요 특징 (Key Features)

* **통합 SDE 인터페이스**: 설정 변경만으로 **VP-SDE** (Variance Preserving)와 **VE-SDE** (Variance Exploding) 방식을 자유롭게 전환할 수 있습니다.
* **학습 안정성 강화**: $t \to 0$ 구간에서의 수치적 불안정을 해결하기 위해, Score를 직접 예측하는 대신 **Noise Prediction ($\epsilon$-prediction)** 방식을 채택했습니다.
* **샘플링 품질 개선**: Sampling 과정에 안전장치(Std Clipping, 마지막 단계 Denoising, Value Clamping)를 추가하여 아티팩트 없는 깨끗한 이미지를 생성합니다.
* **유연한 모델 구조**:
    * **Score U-Net**: Self-Attention과 Gaussian Fourier Projection을 탑재하여 고품질 이미지 생성 (MNIST).
    * **Score MLP**: 2D Manifold 학습을 위한 경량화 모델 (Swiss Roll).

## 📂 프로젝트 구조 (Structure)

```bash
general-sde-diffusion/
├── src/
│   ├── sde/          # 물리 엔진 (VP-SDE, VE-SDE 수식 구현)
│   ├── models/       # 신경망 모델 (Time-dependent U-Net, MLP)
│   └── datasets.py   # 데이터 로더
├── train.py          # 통합 학습 스크립트
├── sample.py         # 샘플링 스크립트 (Euler-Maruyama Solver)
└── requirements.txt  # 필요 라이브러리
```

## 🚀 시작하기 (Getting Started)

### 1. 설치 (Installation)
터미널에서 저장소를 복제(Clone)하고 필수 라이브러리를 설치합니다.

```bash
git clone [https://github.com/Munsik-Kim/general-sde-diffusion.git](https://github.com/Munsik-Kim/general-sde-diffusion.git)
cd general-sde-diffusion
pip install -r requirements.txt
```

2. 학습 (Training)
데이터셋(mnist, swiss_roll)과 SDE 타입(vp, ve)을 선택하여 학습을 진행할 수 있습니다.

예제 1: MNIST + VP-SDE (DDPM 방식) DDPM(Denoising Diffusion Probabilistic Models)과 동일한 방식으로 이미지를 학습합니다.

```Bash

python train.py --dataset mnist --sde_type vp --model_type unet --epochs 50
```
예제 2: Swiss Roll + VE-SDE (Score Matching 방식) 2차원 나선형 데이터 분포를 학습하며 Score Matching의 원리를 확인합니다.

```Bash


python train.py --dataset swiss_roll --sde_type ve --model_type mlp --epochs 500
```

3. 생성 (Sampling)
학습된 모델을 사용하여 새로운 샘플을 생성합니다. 생성된 이미지는 outputs/ 디렉토리에 저장됩니다.

```Bash

python sample.py --dataset mnist --sde_type vp
```

## 📊 구현 상세 (Implementation Details)

본 프로젝트는 두 가지 핵심 SDE 방식을 지원하며, 학습 안정성을 위해 몇 가지 최적화된 기법을 적용했습니다.

### 1. SDE 방식 비교

| 방식 | SDE 타입 | 예측 타겟 | 특징 |
| :--- | :--- | :--- | :--- |
| **DDPM** | VP-SDE | Noise ($\epsilon$) | 데이터의 분산을 고정(Preserving)한 채 점진적으로 노이즈를 주입합니다. |
| **SMLD / NCSN** | VE-SDE | Noise ($\epsilon$) | 데이터의 분산을 폭발적(Exploding)으로 증가시키며 노이즈를 주입합니다. |

### 2. 핵심 기술적 결정 (Technical Decisions)

* **Prediction Target: Noise ($\epsilon$) over Score**
  원 논문(VE-SDE)은 Score 함수($\nabla \log p_t(x)$)를 직접 예측하지만, 본 구현체는 두 방식 모두 **Noise($\epsilon$)를 예측한 뒤 이를 Score로 변환**($\text{Score} \propto -\epsilon / \sigma_t$)하여 사용합니다.
  > **이유**: $t \to 0$ 시점에서 $\sigma_t$가 매우 작아질 때, Score 값이 발산(Exploding)하여 학습이 불안정해지는 문제를 방지하기 위함입니다.

* **Sampling Guardrails**
  고품질 이미지 생성을 위해 샘플링 루프(Euler-Maruyama Solver)에 다음과 같은 수치적 안전장치를 적용했습니다.
  * **Std Clipping**: $\sigma_t$가 0에 가까울 때 나눗셈 오류 방지.
  * **Final Step Denoising**: 마지막 타임스텝에서는 추가 노이즈를 주입하지 않음.
  * **Dynamic Clamping**: 생성 과정에서 픽셀 값이 $[-1, 1]$ 범위를 벗어나지 않도록 제한.

## 📜 References
* Song, Y., et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
