# 💳 Unified Financial Risk Encoder (UFRE): 다중 교사 증류 기반 CSS·FDS 통합 인코딩 아키텍처

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Distillation-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Domain-Credit_Card_Risk-blue?style=flat-square" alt="Credit Card">
</p>

본 프로젝트는 대형 카드사의 **CSS(개인신용평가시스템)** 및 **FDS(이상거래탐지시스템)** 실무 환경의 인프라 병목을 해결하기 위해 기획된 실험적(Exploratory) 모델링 파이프라인입니다. NAVER LABS Europe의 컴퓨터 비전(CV) 연구인 **DIVINE(다중 교사 증류, CVPR 2025/ECCV 2024)** 아키텍처를 금융 리스크 도메인에 이식(Cross-Domain Adaptation)하여 벤치마크를 수행했습니다.

---

## 💡 1. Problem Statement (실무적 배경)

현재 대형 금융사의 데이터 분석 환경은 도메인별로 파이프라인이 파편화(Silo)되어 있습니다.
- **연산 중복 (Computational Redundancy):** 대손 예측(정형 데이터 중심)과 자금세탁/사기 탐지(네트워크 그래프 중심) 모델이 독립적으로 서빙되며, 동일 고객 데이터에 대해 중복된 Feature Extraction 및 I/O 연산이 발생합니다.
- **단편적 시야 (Fragmented Context):** 고객의 '재무적 건전성'과 '거래 네트워크의 건전성'은 상호 보완적인 지표임에도, 개별 모델은 이를 통합된 잠재 공간(Latent Space)에서 이해하지 못합니다.

---

## 🚀 2. Methodology: Attention-based Knowledge Distillation

기존의 단순 앙상블(Ensemble) 기법이 갖는 추론 속도(Inference Latency) 저하 문제를 해결하기 위해, **동적 지식 증류(Dynamic KD)** 기법을 도입했습니다.

<p align="center">
  <img src="distillation_concept.png" width="100%" alt="Architecture">
</p>

### 기술적 구현 상세 (Technical Details)
1. **Teacher Models (전문가 모델):** 
   - `Teacher 1 (CSS)`: Tabular Feature 기반의 MLP 인코더
   - `Teacher 2 (FDS)`: 거래 네트워크 중심성(PageRank 등) 기반 인코더
2. **Student Model (초경량 통합 인코더):**
   - **Cross-Attention Gating Network:** 입력된 고객 패턴을 분석하여, 두 Teacher 중 어떤 모델의 예측값(Soft-label)에 더 높은 신뢰도 가중치를 부여할지 동적으로 학습합니다.
   - **Loss Function:** 실제 정답(Hard-label, BCE Loss)과 교사 모델의 예측 확률(Soft-label, MSE Loss)을 가중 합산하여 학습의 안정성을 확보했습니다.

---

## 📈 3. Quantitative Achievements (정량적 성과)

통합 인코더(Student)는 두 전문가 모델(Teacher) 대비 **더 적은 파라미터로 더 높은 ROC-AUC 성능을 달성**하며, 지식 증류의 비즈니스 효용성을 입증했습니다.

<p align="center">
  <img src="distillation_efficiency_v2.png" width="100%" alt="Efficiency">
</p>

- **성능 시너지 (Synergy Effect):** 상호 이질적인 도메인(정형 vs 그래프)의 지식을 융합하여 단일 모델의 탐지 한계를 극복했습니다.
- **인프라 최적화 (Infrastructure Optimization):** 파라미터 수를 **약 72% 경량화**하여, 실시간(Real-time) 트랜잭션 환경에서 필수적인 API 추론 지연시간(Latency)을 대폭 단축할 수 있는 근거를 마련했습니다.

---

## ⚠️ 4. Limitations & Future Work (한계 및 향후 과제)

본 연구는 알고리즘의 작동 가능성(Feasibility)을 검증하기 위한 PoC(Proof of Concept) 단계로, 다음과 같은 한계점을 갖습니다.
1. **데이터 셋의 한계:** 보안 규정상 실제 카드사 원장 데이터가 아닌, 분포 특성(클래스 불균형, 노이즈)을 모사한 합성 데이터(Synthetic Data)를 사용하여 학습되었습니다. 실제 환경에서는 Data Drift 현상이 발생할 수 있습니다.
2. **향후 개선 방향:** 실제 서빙 환경 도입을 가정하여, ONNX 또는 TensorRT로 모델을 변환하여 T4/A10G GPU 환경에서의 실제 처리량(Throughput) 벤치마크를 추가할 계획입니다.