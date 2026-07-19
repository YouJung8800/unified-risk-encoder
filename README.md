# Unified Financial Risk Encoder (UFRE)

NAVER LABS Europe의 DIVINE(다중 교사 증류, CVPR 2025/ECCV 2024)
개념을 금융 리스크 도메인에 적용한 실험적 구현입니다.

![architecture](distillation_concept.png)

![result](distillation_efficiency.png)

![Python](https://img.shields.io/badge/Python-3.13-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Distillation-orange)

## 핵심 아이디어
DIVINE이 DUSt3R(3D 공간 이해)와 Multi-HMR(인체 인식)이라는 서로 다른
전문 모델의 지식을 하나의 경량 인코더로 압축했듯, 이 프로젝트는
정형 피처 기반 신용리스크 모델과 그래프 기반 AML 모델이라는 두
"교사"의 지식을 하나의 공유 인코더("학생")로 증류합니다.

## 실무적 동기
카드사에서 대손예측, 이상거래탐지, 마케팅 모델이 각각 별도로
서빙되면서 동일 고객 데이터를 중복 인코딩하는 비효율이 발생합니다.
통합 인코더는 이 중복을 줄여 연산 비용과 지연시간을 낮추는 것을
목표로 합니다.

## 결과
파라미터 수와 추론 시간 절감 효과를 정량 측정했습니다.
(정확한 수치는 distillation_results.csv 참고)

## 한계 (정직하게 명시)
- 합성 데이터 기반 검증
- 실제 DIVINE 논문의 3개 이상 태스크·대규모 데이터 실험과 스케일 차이 있음
- 이 구현은 DIVINE의 발명이 아니라, 다중 교사 증류 개념의 금융 도메인 이식 실험

## 실행
\`\`\`bash
pip install torch torch_geometric scikit-learn pandas numpy matplotlib
python3 unified_encoder.py
\`\`\`

## 참고
- NAVER LABS Europe, "DIVINE: 로봇을 위한 유니버설 AI 인코더" (2026)
