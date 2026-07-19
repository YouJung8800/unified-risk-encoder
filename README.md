# Unified Financial Risk Encoder (UFRE)

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Distillation-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

NAVER LABS Europe의 **DIVINE(다중 교사 증류, CVPR 2025/ECCV 2024)** 논문에서 영감을 받아, 최신 컴퓨터 비전(CV) 아키텍처를 금융 리스크 도메인(FDS, CSS)에 최초로 이식(Cross-Domain Application)한 프로젝트입니다.

## 🚀 아키텍처 (Cross-Attention Fusion)
단순히 두 모델의 결괏값을 평균 내는 기존 앙상블의 한계를 극복하기 위해, **Attention 메커니즘을 도입한 동적 지식 증류(Dynamic Knowledge Distillation)**를 구현했습니다.

![Architecture](distillation_concept.png)

## 💡 왜 이게 필요한가 (비즈니스 임팩트)
카드사 및 핀테크 실무에서는 대손예측(정형 데이터)과 이상거래탐지(네트워크 그래프 데이터) 모델이 각각 독립된 파이프라인으로 서빙됩니다. 
통합 인코더(Unified Encoder)는 동일한 고객 데이터를 한 번만 인코딩하여 다음을 달성합니다.
1. **연산 인프라 절감:** 추론(Inference) 단계의 연산량을 대폭 절감하여 API 지연시간(Latency) 단축.
2. **시너지 효과 (성능 향상):** 두 도메인의 지식을 융합하여 단일 모델들을 상회하는 탐지력 확보.

## 📈 결과 대시보드
개별 교사 모델 대비 통합 인코더의 **ROC-AUC 향상**과, **파라미터 절감률**을 정량 측정했습니다.
![Result](distillation_efficiency_v2.png)

## 🛠 주요 방법론 고도화 (이전 버전 대비)
- **Attention-based Knowledge Distillation:** Student 모델 내부에 Attention Gate를 두어, 특정 고객의 거래 패턴을 분석할 때 '정형 데이터'와 '그래프 데이터' 중 어떤 교사의 예측을 더 신뢰할지 동적으로 가중치를 부여하여 증류받습니다.
- **실무 수준의 데이터 모사:** 클래스 불균형과 노이즈를 반영하여 베이스라인 성능을 실무 수준(AUC 0.78 이상)으로 스케일업 하였습니다. 
