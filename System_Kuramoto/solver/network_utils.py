"""
Network Utilities for Kuramoto Oscillator
==========================================
Explosive Synchronization(ES)에 필요한 네트워크 생성 및 분석 도구.

핵심 아이디어 (Gómez-Gardeñes et al., PRL 2011):
    - 척도 없는 네트워크 (Barabási-Albert, BA): 허브 노드가 존재
    - 차수-주파수 상관: ω_i = k_i  (차수가 클수록 고유 진동수도 큼)
    → 결합 강도 K 증가 시 비연속적 점프(1차 전이) + 히스테리시스 발생

일반 전결합 쿠라모토는 2차 전이(연속)이지만
BA 네트워크 + ω_i = k_i 조합이면 1차 전이(폭발적 동기화)가 나타남.
"""

import numpy as np
import networkx as nx


# ──────────────────────────────────────────────
#  네트워크 생성
# ──────────────────────────────────────────────

def make_ba_network(N: int, m: int = 2, seed: int = 42) -> nx.Graph:
    """Barabási-Albert scale-free 네트워크 생성.

    Parameters
    ----------
    N : int
        노드 수.
    m : int
        각 신규 노드가 연결하는 간선 수 (차수 분포 : P(k) ~ k^{-3}).
    seed : int
        재현성을 위한 난수 시드.

    Returns
    -------
    G : nx.Graph
    """
    return nx.barabasi_albert_graph(N, m, seed=seed)


def make_erdos_renyi_network(N: int, p: float = 0.1, seed: int = 42) -> nx.Graph:
    """Erdős-Rényi 무작위 네트워크 (비교 기준용)."""
    return nx.erdos_renyi_graph(N, p, seed=seed)


def make_complete_network(N: int) -> nx.Graph:
    """전결합 네트워크 (기존 표준 쿠라모토 모델)."""
    return nx.complete_graph(N)


# ──────────────────────────────────────────────
#  Explosive Synchronization 전용 주파수 할당
# ──────────────────────────────────────────────

def assign_frequencies_es(G: nx.Graph) -> np.ndarray:
    """ES 조건: ω_i = k_i  (차수-주파수 양의 상관, 원본 스케일).

    주의: k_i의 std가 크면 K_c도 매우 커짐 (BA N=100,m=3: std≈5.2, K_c≈3.3).
    실험 목적이면 assign_frequencies_es_normalized() 사용 권장.
    """
    degrees = np.array([d for _, d in G.degree()])
    return degrees.astype(float)


def assign_frequencies_es_normalized(G: nx.Graph) -> np.ndarray:
    """ES 조건 (정규화): ω_i = k_i / <k>.

    평균 차수로 나눠 주파수 스케일을 O(1)로 맞춘다.
    - std(ω) ≈ std(k)/<k> ≈ 0.9  (BA N=100, m=3)
    - 추정 K_c ≈ 2*std(ω)/π ≈ 0.57   → K_MAX = 2.0 이면 충분

    Gómez-Gardeñes (2011) 논문의 실질적 구현에서도 이 수준의 스케일을 사용.
    """
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    mean_k  = degrees.mean()
    return degrees / mean_k   # ω_i = k_i / <k>


def assign_frequencies_random(G: nx.Graph, std: float = 1.0,
                               seed: int = 42) -> np.ndarray:
    """비상관 랜덤 주파수 (일반 쿠라모토 비교용)."""
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    return rng.normal(0.0, std, size=N)


# ──────────────────────────────────────────────
#  네트워크 행렬 추출
# ──────────────────────────────────────────────

def get_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """인접 행렬 (dense numpy array)."""
    return nx.to_numpy_array(G, dtype=float)


def get_degree_array(G: nx.Graph) -> np.ndarray:
    """각 노드의 차수 벡터."""
    return np.array([d for _, d in G.degree()], dtype=float)


# ──────────────────────────────────────────────
#  네트워크 통계 요약
# ──────────────────────────────────────────────

def network_summary(G: nx.Graph) -> dict:
    """네트워크 기본 통계 반환."""
    degrees = np.array([d for _, d in G.degree()])
    return {
        "N":             G.number_of_nodes(),
        "edges":         G.number_of_edges(),
        "mean_degree":   degrees.mean(),
        "max_degree":    degrees.max(),
        "min_degree":    degrees.min(),
        "std_degree":    degrees.std(),
        "is_connected":  nx.is_connected(G),
    }


# ──────────────────────────────────────────────
#  테스트
# ──────────────────────────────────────────────
