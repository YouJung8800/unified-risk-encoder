"""
Unified Financial Risk Encoder (UFRE)
NAVER LABS Europe의 DIVINE(다중 교사 증류) 개념을 금융 리스크 도메인에 적용.

핵심 아이디어: 개별 태스크(대손예측, 이상거래탐지)마다 별도 인코더를 쓰는
비효율을 없애고, 하나의 공유 인코더가 두 교사 모델의 지식을 증류받아
압축된 표현을 학습하도록 설계.

Teacher 1: 신용리스크 MLP (정형 피처 기반)
Teacher 2: AML GAT (그래프 기반)
Student: 하나의 경량 공유 인코더 (Unified Encoder)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(42)
np.random.seed(42)

# ================================================================
# 1. 합성 데이터: 고객 단위로 "정형 리스크 신호"와 "네트워크 리스크 신호"를 동시 보유
# ================================================================
N = 3000
FEAT_DIM_TABULAR = 8   # 신용리스크 정형 피처 (LIMIT_BAL, PAY_0 등)
FEAT_DIM_GRAPH = 5      # 그래프 기반 피처 (PageRank, Degree 등, 사전 계산되었다 가정)

X_tabular = np.random.randn(N, FEAT_DIM_TABULAR).astype(np.float32)
X_graph = np.random.randn(N, FEAT_DIM_GRAPH).astype(np.float32)

# 두 신호가 서로 다른 관점에서 같은 리스크를 부분적으로 설명하도록 라벨 생성
risk_score = (X_tabular[:, 0] * 0.4 + X_tabular[:, 1] * 0.3
              + X_graph[:, 0] * 0.5 + X_graph[:, 1] * 0.2)
prob = 1 / (1 + np.exp(-risk_score))
y = (np.random.rand(N) < prob).astype(np.int64)

print(f"데이터 생성 완료: {N}건, 이상 비율 {y.mean():.2%}")

X_tab_train, X_tab_test, X_graph_train, X_graph_test, y_train, y_test = train_test_split(
    X_tabular, X_graph, y, test_size=0.25, random_state=42, stratify=y)

# ================================================================
# 2. Teacher 모델 1: 정형 피처 기반 리스크 모델 (기존 credit-risk 프로젝트에 대응)
# ================================================================
class TabularTeacher(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Linear(16, 2)
    def forward(self, x):
        emb = self.net(x)
        return emb, self.head(emb)

# ================================================================
# 3. Teacher 모델 2: 그래프 피처 기반 리스크 모델 (기존 spatial-aml-mapping에 대응)
# ================================================================
class GraphTeacher(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU())
        self.head = nn.Linear(16, 2)
    def forward(self, x):
        emb = self.net(x)
        return emb, self.head(emb)

# ================================================================
# 4. Student 모델: DIVINE의 "유니버설 인코더"에 해당
#    두 교사의 서로 다른 입력을 하나의 공유 표현 공간으로 압축
# ================================================================
class UnifiedRiskEncoder(nn.Module):
    """
    두 개의 서로 다른 태스크(정형 피처, 그래프 피처)를 처리하던 개별
    인코더를 하나로 통합. DIVINE이 DUSt3R와 Multi-HMR을 하나의
    인코더로 압축한 것과 동일한 설계 철학.
    """
    def __init__(self, tab_dim, graph_dim, shared_dim=16):
        super().__init__()
        # 입력 어댑터: 서로 다른 차원의 입력을 공통 공간으로 투영
        self.tab_adapter = nn.Linear(tab_dim, shared_dim)
        self.graph_adapter = nn.Linear(graph_dim, shared_dim)
        # 공유 인코더 (핵심: 하나의 백본만 존재)
        self.shared_encoder = nn.Sequential(
            nn.Linear(shared_dim, 24), nn.ReLU(),
            nn.Linear(24, 16), nn.ReLU())
        self.risk_head = nn.Linear(16, 2)

    def forward(self, x_tab, x_graph):
        h_tab = self.tab_adapter(x_tab)
        h_graph = self.graph_adapter(x_graph)
        h_fused = (h_tab + h_graph) / 2  # 두 신호 융합
        emb = self.shared_encoder(h_fused)
        return emb, self.risk_head(emb)

# ================================================================
# 5. 학습: 먼저 두 교사를 개별 학습, 그 다음 학생이 지식 증류받음
# ================================================================
X_tab_train_t = torch.tensor(X_tab_train)
X_graph_train_t = torch.tensor(X_graph_train)
y_train_t = torch.tensor(y_train)
X_tab_test_t = torch.tensor(X_tab_test)
X_graph_test_t = torch.tensor(X_graph_test)

teacher_tab = TabularTeacher(FEAT_DIM_TABULAR)
teacher_graph = GraphTeacher(FEAT_DIM_GRAPH)

opt_tab = torch.optim.Adam(teacher_tab.parameters(), lr=0.01)
opt_graph = torch.optim.Adam(teacher_graph.parameters(), lr=0.01)

print("\n=== Teacher 모델 학습 ===")
for epoch in range(100):
    opt_tab.zero_grad()
    _, out_tab = teacher_tab(X_tab_train_t)
    loss_tab = F.cross_entropy(out_tab, y_train_t)
    loss_tab.backward()
    opt_tab.step()

    opt_graph.zero_grad()
    _, out_graph = teacher_graph(X_graph_train_t)
    loss_graph = F.cross_entropy(out_graph, y_train_t)
    loss_graph.backward()
    opt_graph.step()

teacher_tab.eval()
teacher_graph.eval()
with torch.no_grad():
    prob_tab = F.softmax(teacher_tab(X_tab_test_t)[1], dim=1)[:, 1]
    prob_graph = F.softmax(teacher_graph(X_graph_test_t)[1], dim=1)[:, 1]
auc_tab = roc_auc_score(y_test, prob_tab.numpy())
auc_graph = roc_auc_score(y_test, prob_graph.numpy())
print(f"Teacher 1 (정형피처) ROC-AUC: {auc_tab:.4f}")
print(f"Teacher 2 (그래프피처) ROC-AUC: {auc_graph:.4f}")

# ================================================================
# 6. Student 학습: 다중 교사 증류 (Multi-Teacher Distillation)
# ================================================================
student = UnifiedRiskEncoder(FEAT_DIM_TABULAR, FEAT_DIM_GRAPH)
opt_student = torch.optim.Adam(student.parameters(), lr=0.01)

T = 3.0  # distillation temperature
alpha = 0.5  # 증류손실과 실제라벨손실의 가중치

print("\n=== Student(Unified Encoder) 다중 교사 증류 학습 ===")
for epoch in range(150):
    student.train()
    opt_student.zero_grad()

    _, student_out = student(X_tab_train_t, X_graph_train_t)

    with torch.no_grad():
        _, teacher_tab_out = teacher_tab(X_tab_train_t)
        _, teacher_graph_out = teacher_graph(X_graph_train_t)
        # 두 교사의 soft label을 평균 (다중 교사 증류의 핵심)
        teacher_soft = (F.softmax(teacher_tab_out / T, dim=1) +
                         F.softmax(teacher_graph_out / T, dim=1)) / 2

    distill_loss = F.kl_div(F.log_softmax(student_out / T, dim=1),
                              teacher_soft, reduction='batchmean') * (T * T)
    label_loss = F.cross_entropy(student_out, y_train_t)
    loss = alpha * distill_loss + (1 - alpha) * label_loss

    loss.backward()
    opt_student.step()

    if epoch % 30 == 0:
        print(f"  epoch {epoch:3d} | distill_loss: {distill_loss.item():.4f} | "
              f"label_loss: {label_loss.item():.4f}")

student.eval()
with torch.no_grad():
    _, student_test_out = student(X_tab_test_t, X_graph_test_t)
    prob_student = F.softmax(student_test_out, dim=1)[:, 1]
auc_student = roc_auc_score(y_test, prob_student.numpy())
print(f"\nStudent(통합인코더) ROC-AUC: {auc_student:.4f}")

# ================================================================
# 7. DIVINE 논문이 제시한 핵심 지표 재현: 연산 효율 비교
# ================================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def measure_inference_time(fn, *args, n_trials=100):
    start = time.time()
    for _ in range(n_trials):
        with torch.no_grad():
            fn(*args)
    return (time.time() - start) / n_trials * 1000  # ms

params_separate = count_params(teacher_tab) + count_params(teacher_graph)
params_unified = count_params(student)

time_separate = (measure_inference_time(teacher_tab, X_tab_test_t) +
                  measure_inference_time(teacher_graph, X_graph_test_t))
time_unified = measure_inference_time(student, X_tab_test_t, X_graph_test_t)

param_reduction = (1 - params_unified / params_separate) * 100
time_reduction = (1 - time_unified / time_separate) * 100

print(f"\n=== 효율성 비교 (DIVINE 논문의 핵심 주장 재현) ===")
print(f"개별 모델 파라미터 수: {params_separate:,} vs 통합 인코더: {params_unified:,} "
      f"({param_reduction:+.1f}%)")
print(f"개별 모델 추론 시간: {time_separate:.3f}ms vs 통합 인코더: {time_unified:.3f}ms "
      f"({time_reduction:+.1f}%)")

# ================================================================
# 8. 결과 저장 및 시각화
# ================================================================
results = pd.DataFrame({
    "model": ["Teacher-정형", "Teacher-그래프", "Student-통합인코더"],
    "roc_auc": [auc_tab, auc_graph, auc_student],
    "params": [count_params(teacher_tab), count_params(teacher_graph), params_unified]
})
results.to_csv("distillation_results.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(results["model"], results["roc_auc"], color=["gray", "gray", "steelblue"])
axes[0].set_title("모델별 ROC-AUC 비교")
axes[0].set_ylim(0, 1)
for i, v in enumerate(results["roc_auc"]):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center")

axes[1].bar(["개별 모델 합", "통합 인코더"], [params_separate, params_unified],
            color=["salmon", "steelblue"])
axes[1].set_title(f"파라미터 수 비교 (통합 시 {param_reduction:.1f}% 절감)")
axes[1].set_ylabel("파라미터 수")

plt.tight_layout()
plt.savefig("distillation_efficiency.png", dpi=150)
print("\n시각화 저장: distillation_efficiency.png")

torch.save(student.state_dict(), "unified_encoder.pt")
print("모델 저장 완료: unified_encoder.pt")
