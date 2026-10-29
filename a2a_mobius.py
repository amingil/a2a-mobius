"""
═══════════════════════════════════════════════════════════════════════════════
 a2a_mobius.py — Monolito modular A2A heterogéneo con capa Möbius-fractal
═══════════════════════════════════════════════════════════════════════════════

Filosofía del archivo
─────────────────────
Cada bloque distingue propiedades algebraicas verificables, heurísticas y
límites. Este MVP no demuestra una banda de Möbius, una fase de Berry ni
alineación semántica entre modelos.

Flujo end-to-end
────────────────
  texto → embedder(A) → Bridge(A→B) → Möbius(B) → fractal_avg(B)
                                             → DecodeInterface + Task versionado

Adaptación
──────────
  Nivel 0: mapeo de prefijos Matryoshka, sólo si el encoder garantiza ejes anidados
  Nivel 1: Procrustes z/log, seleccionado con anclas reservadas
  Nivel 2: Kernel Ridge, sólo si mejora validación por al menos 20%
Dimensiones grandes: puente Procrustes de bajo rango en el espacio de anclas;
                    no materializa la matriz cruzada dA×dB.

El transporte HTTP incluido es un JSON-RPC experimental local con una parte
tensorial propia; no se anuncia como implementación conforme al estándar A2A.
UDP modifica una matriz de Gram y se limita a corpus pequeños; la factorización
de rango reducido puede romper las restricciones y el reporte lo mide.
"""

from __future__ import annotations
import asyncio, base64, difflib, hashlib, heapq, hmac, inspect, ipaddress, json, math, os, secrets, sqlite3, ssl, tempfile, threading, time, unicodedata, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.request

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 0. UTILIDADES NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════════════

def skew(A: np.ndarray) -> np.ndarray:
    """Parte antisimétrica de A."""
    return A - A.T


def expm_skew(A: np.ndarray) -> np.ndarray:
    """
    exp(A) para A skew-simétrica → matriz en SO(n).
    Implementación sin scipy: usa eigendecomposición de iA (Hermitiana).
    GARANTÍA: resultado ∈ SO(n), es decir, RᵀR = I y det(R) = +1.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.all(np.isfinite(A)):
        raise ValueError("A debe ser una matriz cuadrada finita.")
    if not np.allclose(A + A.T, 0.0, atol=1e-12, rtol=1e-12):
        raise ValueError("A debe ser antisimétrica: Aᵀ=-A.")
    # iA es Hermitiana cuando A es real y antisimétrica. Su EVD calcula
    # exp(A) sin aproximar por truncamiento de Taylor.
    H = 1j * A
    w, V = np.linalg.eigh(H)
    result = np.real(V @ np.diag(np.exp(-1j * w)) @ V.conj().T)
    # Evita una segunda SVD cuando el redondeo ya conserva la ortogonalidad.
    if ortho_drift(result) > 1e-10 or np.linalg.det(result) < 0.0:
        result = polar_orthogonal(result, special=True)
    return result


def random_so(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Matriz ortogonal especial aleatoria de SO(n)."""
    rng = np.random.default_rng(seed)
    return expm_skew(skew(rng.standard_normal((n, n))))


def qr_twice(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    QR con re-ortogonalización (dos pasadas).
    La segunda pasada corrige deriva numérica del primer factor Q. Se exige
    rango columna completo: ningún QR puede recuperar columnas dependientes.
    USO: construir C del conector Möbius y comprobar la geometría resultante.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] < A.shape[1] or A.shape[1] == 0:
        raise ValueError("A debe tener forma (m,n), con m >= n > 0.")
    if not np.all(np.isfinite(A)):
        raise ValueError("A contiene NaN o infinito.")
    singular_values = np.linalg.svd(A, compute_uv=False)
    threshold = max(A.shape) * np.finfo(np.float64).eps * singular_values[0]
    if np.any(singular_values <= threshold):
        raise ValueError("A no tiene rango columna completo; QR no puede recuperar esa información.")
    Q1, R1 = np.linalg.qr(A, mode="reduced")
    Q2, R2 = np.linalg.qr(Q1, mode="reduced")
    return Q2, R2 @ R1


def polar_orthogonal(M: np.ndarray, *, special: bool = False) -> np.ndarray:
    """
    Matriz ortogonal MÁS CERCANA a M en norma de Frobenius.
    Diferencia con QR: QR devuelve "una" ortonormal; polar devuelve la más
    cercana, preservando la orientación. Ideal para corregir DERIVA de R
    acumulada en muchas interacciones.
    GARANTÍA: ‖M_polar − M‖_F es mínima entre todas las ortogonales.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or not np.all(np.isfinite(M)):
        raise ValueError("M debe ser una matriz cuadrada finita.")
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    if special and np.linalg.det(U @ Vt) < 0.0:
        U[:, -1] *= -1.0
    return U @ Vt


def ortho_drift(M: np.ndarray) -> float:
    """‖MᵀM − I‖_F: mide cuánto se alejó M de ser ortogonal."""
    return float(np.linalg.norm(M.T @ M - np.eye(M.shape[1]), 'fro'))


def _randomized_svd(matrix: np.ndarray, n_components: int,
                    n_oversamples: int = 10, n_iter: int = 4,
                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula una SVD truncada mediante rango aleatorio e iteración de potencia.

    Devuelve U, S y Vt con las primeras `n_components` componentes. Es una
    aproximación determinista para una semilla fija; si no hay truncamiento
    posible, delega en la SVD completa para preservar el resultado exacto.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape, default=0) < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("matrix debe ser una matriz 2D finita y no vacía.")
    if (not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool)
            or not 1 <= n_components <= min(matrix.shape)):
        raise ValueError("n_components debe estar entre 1 y min(matrix.shape).")
    if (not isinstance(n_oversamples, (int, np.integer)) or isinstance(n_oversamples, bool)
            or n_oversamples < 0):
        raise ValueError("n_oversamples debe ser un entero no negativo.")
    if not isinstance(n_iter, (int, np.integer)) or isinstance(n_iter, bool) or n_iter < 0:
        raise ValueError("n_iter debe ser un entero no negativo.")

    m, n = matrix.shape
    k = int(n_components)
    rank_limit = min(k + int(n_oversamples), m, n)
    if rank_limit == min(m, n):
        U, singular, Vt = np.linalg.svd(matrix, full_matrices=False)
        return U[:, :k], singular[:k], Vt[:k]

    rng = np.random.default_rng(seed)
    sample = matrix @ rng.standard_normal((n, rank_limit))
    for _ in range(int(n_iter)):
        Q, _ = np.linalg.qr(sample, mode="reduced")
        Z, _ = np.linalg.qr(matrix.T @ Q, mode="reduced")
        sample = matrix @ Z
    Q, _ = np.linalg.qr(sample, mode="reduced")
    small = Q.T @ matrix
    Ub, singular, Vt = np.linalg.svd(small, full_matrices=False)
    return Q @ Ub[:, :k], singular[:k], Vt[:k]


# ═══════════════════════════════════════════════════════════════════════════
# 1. EMBEDDERS (Nivel 0 / Nivel 1 / interfaz Nivel 2)
# ═══════════════════════════════════════════════════════════════════════════

def dhash(s: str, salt: int = 0) -> int:
    """Hash determinista (a diferencia de hash() de Python, reproducible entre procesos)."""
    return int(hashlib.sha256(f"{salt}:{s}".encode()).hexdigest()[:8], 16)


_ARABIC_NORMALIZATION_TABLE = str.maketrans({
    **{codepoint: None for codepoint in (
        *range(0x0610, 0x061B), *range(0x064B, 0x0660), *range(0x06D6, 0x06EE))},
    ord("إ"): "ا", ord("أ"): "ا", ord("آ"): "ا", ord("ٱ"): "ا",
    ord("ى"): "ي", ord("ة"): "ه",
})


def ar_norm(t: str) -> str:
    """
    Normalización árabe canónica: quita tashkeel, unifica alef/hamza/ya/ta-marbuta.
    CRÍTICO: sin esto, "لَا" y "لا" producen embeddings distintos.
    """
    t = unicodedata.normalize('NFKC', t)
    return ' '.join(t.translate(_ARABIC_NORMALIZATION_TABLE).split())


class TrigramEmbedder:
    """
    Embedder determinista de trigramas (Nivel 1, para smoke/unit tests).
    NO es un LLM; sirve para reproducibilidad y CI.
    Reemplazable por GTE-Qwen2/E5 en producción.
    """
    def __init__(self, dim: int = 128, salt: int = 0):
        self.dim, self.salt = dim, salt

    def __call__(self, text: str) -> np.ndarray:
        t = ar_norm(text)
        v = np.zeros(self.dim)
        for i in range(max(0, len(t) - 2)):
            v[dhash(t[i:i+3], self.salt) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class SharedBasisEmbedder:
    """
    Embedder sintético para comprobar un caso de alineación conocido.

    Ambos agentes construyen el mismo vector base de trigramas y lo insertan
    mediante un marco ortonormal distinto. Esto crea una relación lineal
    recuperable en el subespacio compartido. Es un control geométrico, no una
    representación semántica ni evidencia de alineación entre LLM reales.
    """
    def __init__(self, trigrams: List[str], dim: int, seed: int = 0,
                 basis_dim: int = 12):
        if dim < 1 or basis_dim < 1 or not trigrams:
            raise ValueError("dim, basis_dim y la lista de trigramas deben ser positivos.")
        self.tri2idx = {tg: i for i, tg in enumerate(trigrams)}
        self.T = len(trigrams)
        self.dim = int(dim)
        self.basis_dim = min(int(basis_dim), self.dim)
        rng = np.random.default_rng(seed)
        raw_frame = rng.standard_normal((self.dim, self.basis_dim))
        self.R, _ = qr_twice(raw_frame)

    def __call__(self, text: str) -> np.ndarray:
        x = np.zeros(self.basis_dim)
        t = ar_norm(text)
        for i in range(max(0, len(t) - 2)):
            tg = t[i:i+3]
            if tg in self.tri2idx:
                x[dhash(tg) % self.basis_dim] += 1.0
        v = self.R @ x
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


# ═══════════════════════════════════════════════════════════════════════════
# 2. UDP EMBEDDER (proyección de Gram con auditoría posterior)
# ═══════════════════════════════════════════════════════════════════════════

class UDPEmbedder:
    """
    Proyecta un Gram pequeño hacia diag(G)=1 y G[i,j]=0.5 en aristas UDP.

    Para vectores unitarios, G[i,j]=0.5 equivale a distancia euclidiana 1.
    Se usa Dykstra entre el cono PSD y el subespacio afín de restricciones.
    La factorización con rango limitado y la renormalización posteriores
    pueden romper las restricciones; `report` mide esa pérdida y no la oculta.
    El cálculo es O(N^3) por iteración y está pensado para prototipos pequeños.
    """
    def __init__(self, base_embedder: Callable[[str], np.ndarray],
                 udp_edges: List[Tuple[int, int]],
                 output_dim: int | None = None,
                 n_iter: int = 500, tol: float = 1e-8,
                 max_points: int = 512):
        if n_iter < 1 or not math.isfinite(tol) or tol <= 0 or max_points < 1:
            raise ValueError("n_iter, tol y max_points deben ser positivos.")
        self.base = base_embedder
        self.edges = [(int(i), int(j)) for i, j in udp_edges]
        self.output_dim = output_dim
        self.n_iter, self.tol, self.max_points = n_iter, tol, max_points
        self.X_udp: Optional[np.ndarray] = None
        self.report: Dict = {}
        self._corpus_index: Dict[str, int] = {}

    def fit(self, corpus: List[str]) -> 'UDPEmbedder':
        if not corpus or len(corpus) > self.max_points:
            raise ValueError(f"corpus debe tener entre 1 y {self.max_points} filas.")
        X = np.asarray([self.base(t) for t in corpus], dtype=np.float64)
        if X.ndim != 2 or not np.all(np.isfinite(X)):
            raise ValueError("El embedder debe producir una matriz finita.")
        norms = np.linalg.norm(X, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("UDP no admite vectores base de norma cero.")
        X = X / norms[:, None]
        n = len(X)
        if not self.edges:
            raise ValueError("Hace falta al menos una arista; sin aristas no hay invariante UDP que auditar.")
        for i, j in self.edges:
            if not (0 <= i < n and 0 <= j < n) or i == j:
                raise ValueError(f"Arista UDP inválida: {(i, j)} para N={n}.")

        gram0 = X @ X.T
        gram = gram0.copy()
        correction_psd = np.zeros_like(gram)
        correction_affine = np.zeros_like(gram)
        target_inner = 0.5  # cuerda euclidiana de longitud uno en la esfera unidad
        converged = False
        iterations = 0
        for iterations in range(1, self.n_iter + 1):
            previous = gram.copy()
            residual = gram - correction_psd
            symmetric = (residual + residual.T) * 0.5
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
            clipped = np.maximum(eigenvalues, 0.0)
            psd = (eigenvectors * clipped) @ eigenvectors.T
            correction_psd = psd - residual

            residual = psd - correction_affine
            affine = (residual + residual.T) * 0.5
            np.fill_diagonal(affine, 1.0)
            for i, j in self.edges:
                affine[i, j] = affine[j, i] = target_inner
            correction_affine = affine - residual
            gram = affine
            change = float(np.linalg.norm(gram - previous, ord="fro"))
            if change <= self.tol * max(1.0, float(np.linalg.norm(gram, ord="fro"))):
                converged = True
                break

        # Factorizar la matriz PSD final y limitar rango si se pidió una
        # dimensión menor. La auditoría se realiza sobre las coordenadas finales.
        gram = (gram + gram.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        positive = order[eigenvalues[order] > self.tol]
        wanted = X.shape[1] if self.output_dim is None else self.output_dim
        if not isinstance(wanted, (int, np.integer)) or wanted < 1 or wanted > 12288:
            raise ValueError("output_dim debe estar entre 1 y D_max=12288.")
        selected = positive[:wanted]
        coords = eigenvectors[:, selected] * np.sqrt(eigenvalues[selected])[None, :]
        if coords.shape[1] < wanted:
            coords = np.pad(coords, ((0, 0), (0, wanted - coords.shape[1])))
        pre_normalization_norms = np.linalg.norm(coords, axis=1)
        if np.any(pre_normalization_norms <= 1e-12):
            raise ArithmeticError("La proyección produjo un punto de norma cero.")
        coords = coords / pre_normalization_norms[:, None]
        final_norms = np.linalg.norm(coords, axis=1)
        edge_distances = np.asarray([np.linalg.norm(coords[i] - coords[j])
                                     for i, j in self.edges])
        gram_final = coords @ coords.T
        edge_error = float(np.max(np.abs(edge_distances - 1.0)))
        self.X_udp = coords
        self._corpus_index = {text: index for index, text in enumerate(corpus)}
        self.report = {
            "converged": converged,
            "iterations": iterations,
            "max_edge_error": edge_error,
            "max_norm_error": float(np.max(np.abs(final_norms - 1.0))),
            "pre_normalization_max_norm_deviation": float(
                np.max(np.abs(pre_normalization_norms - 1.0))),
            "relative_gram_distortion": float(
                np.linalg.norm(gram_final - gram0, ord="fro") /
                max(np.linalg.norm(gram0, ord="fro"), 1e-12)),
            "rank_after_projection": int(len(selected)),
            "output_dim": int(wanted),
            "accepted": bool(converged and edge_error <= max(10 * self.tol, 1e-6)),
        }
        return self

    def embed(self, text: str) -> np.ndarray:
        if self.X_udp is None:
            raise RuntimeError("UDPEmbedder.fit debe ejecutarse antes de embed.")
        if text in self._corpus_index:
            return self.X_udp[self._corpus_index[text]].copy()
        # No se proyectó una consulta fuera del grafo de entrenamiento.
        vector = np.asarray(self.base(text), dtype=np.float64)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise ValueError("El embedder devolvió un vector de norma cero.")
        vector = vector / norm
        if vector.size != self.X_udp.shape[1]:
            raise ValueError("Consulta no entrenada: su dimensión base no coincide con output_dim.")
        return vector.copy()


def verify_udp_invariant(X_src: np.ndarray, X_aligned: np.ndarray,
                         udp_edges: List[Tuple[int, int]],
                         alpha_src: float = 1.0, alpha_tgt: float = 1.0,
                         tol: float = 1e-6) -> Dict:
    """
    Audita aristas tras un puente; es una comprobación numérica, no una
    certificación formal. `alpha_*` expresa el factor de escala declarado.
    """
    source, aligned = np.asarray(X_src), np.asarray(X_aligned)
    if (source.ndim != 2 or aligned.ndim != 2 or len(source) != len(aligned)
            or not np.all(np.isfinite(source)) or not np.all(np.isfinite(aligned))):
        raise ValueError("Los estados deben ser matrices con las mismas filas pareadas.")
    if not udp_edges:
        raise ValueError("Hace falta al menos una arista para auditar.")
    if (not math.isfinite(alpha_src) or not math.isfinite(alpha_tgt)
            or not math.isfinite(tol) or abs(alpha_src) <= 1e-12 or tol < 0):
        raise ValueError("alpha_src, alpha_tgt y tol deben ser finitos; alpha_src no puede ser cero.")
    n = len(source)
    if any(i < 0 or j < 0 or i >= n or j >= n or i == j for i, j in udp_edges):
        raise ValueError("Las aristas UDP deben conectar dos filas distintas existentes.")
    scale = alpha_tgt / max(abs(alpha_src), 1e-12)
    source_distances = np.asarray([np.linalg.norm(source[i] - source[j])
                                   for i, j in udp_edges])
    aligned_distances = np.asarray([np.linalg.norm(aligned[i] - aligned[j])
                                    for i, j in udp_edges])
    expected = scale * source_distances
    distortion = np.abs(aligned_distances - expected)
    return {"max_err_source_to_target": float(distortion.max()),
            "max_udp_target_error": float(np.max(np.abs(aligned_distances - alpha_tgt))),
            "preserved": bool(distortion.max() <= tol),
            "udp_target_satisfied": bool(np.max(np.abs(aligned_distances - alpha_tgt)) <= tol)}


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONECTOR MÖBIUS (QR doble + rotor lift)
# ═══════════════════════════════════════════════════════════════════════════

class MobiusConnector:
    """
    T = C · R · Cᵀ + (I − C·Cᵀ)
    donde C ∈ Stiefel(d, d') con CᵀC = I (garantizado por QR doble) y
    R ∈ SO(d') es un rotor de 6D.

    En modo pca/hybrid, `pca_solver="randomized"` aproxima sólo los primeros
    componentes solicitados; usa `pca_solver="full"` para reproducir la SVD
    densa completa.

    TEOREMA: Tᵀ T = I_d PARA CUALQUIER R ∈ SO(d'). Es isometría, no denoiser.
    CONSECUENCIA: ‖T·v‖ = ‖v‖. La información no colapsa por construcción.
    AXIOMA FALSO: "T elimina ruido" — imposible: es isometría.
    """
    def __init__(self, d: int, d_prime: int = 6, mode: str = "hybrid", seed: int = 0,
                 pca_solver: str = "randomized", n_oversamples: int = 10,
                 n_iter: int = 4):
        if d < 2 or d_prime < 2 or d_prime > d:
            raise ValueError("Se requiere d >= d_prime >= 2.")
        if mode not in {"random", "pca", "hybrid"}:
            raise ValueError(f"modo desconocido: {mode}")
        if pca_solver not in {"randomized", "full"}:
            raise ValueError("pca_solver debe ser 'randomized' o 'full'.")
        if (not isinstance(n_oversamples, (int, np.integer)) or isinstance(n_oversamples, bool)
                or n_oversamples < 0 or not isinstance(n_iter, (int, np.integer))
                or isinstance(n_iter, bool) or n_iter < 0):
            raise ValueError("n_oversamples y n_iter deben ser enteros no negativos.")
        self.d, self.d_prime, self.mode, self.seed = d, d_prime, mode, seed
        self.pca_solver = pca_solver
        self.n_oversamples, self.n_iter = int(n_oversamples), int(n_iter)
        self._eye_dp = np.eye(self.d_prime, dtype=np.float64)
        self._eye_dp.setflags(write=False)
        self.C: Optional[np.ndarray] = None

    def _principal_vt(self, Xc: np.ndarray, n_components: int) -> np.ndarray:
        if self.pca_solver == "full":
            return np.linalg.svd(Xc, full_matrices=False)[2]
        return _randomized_svd(
            Xc, n_components, n_oversamples=self.n_oversamples,
            n_iter=self.n_iter, seed=self.seed)[2]

    def fit(self, X: np.ndarray) -> 'MobiusConnector':
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d or len(X) == 0:
            raise ValueError("X debe ser una matriz de anclas con dimensión d.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contiene NaN o infinito.")
        if self.mode == "random":
            C_raw = np.random.default_rng(self.seed).standard_normal(
                (self.d, self.d_prime))
        elif self.mode == "pca":
            Xc = X - X.mean(0)
            available = min(Xc.shape)
            if available < self.d_prime:
                raise ValueError(
                    f"PCA sólo produjo {available} componentes; se requieren {self.d_prime}. "
                    "Aporta más anclas o usa mode='hybrid'.")
            Vt = self._principal_vt(Xc, self.d_prime)
            if len(Vt) < self.d_prime:
                raise ValueError(
                    f"PCA sólo produjo {len(Vt)} componentes; se requieren {self.d_prime}. "
                    "Aporta más anclas o usa mode='hybrid'.")
            C_raw = Vt[:self.d_prime].T
        elif self.mode == "hybrid":
            k_pca = self.d_prime // 2
            Xc = X - X.mean(0)
            available = min(k_pca, min(Xc.shape))
            Vt = self._principal_vt(Xc, available)
            V_pca = Vt[:available].T
            random_columns = self.d_prime - V_pca.shape[1]
            V_rand = np.random.default_rng(self.seed).standard_normal(
                (self.d, random_columns))
            V_rand = V_rand - V_pca @ (V_pca.T @ V_rand)
            C_raw = np.hstack([V_pca, V_rand])
        else:
            raise ValueError(f"modo desconocido: {self.mode}")
        self.C, _ = qr_twice(C_raw)   # ← garantiza CᵀC = I a precisión de máquina
        return self

    def fit_identity(self) -> 'MobiusConnector':
        """Configura un subespacio canónico para flujos latentes sin encoder.

        No aprende una representación semántica: permite usar el protocolo
        con vectores externos sin exigir corpus, embeddings ni entrenamiento.
        """
        self.C = np.eye(self.d, self.d_prime, dtype=np.float64)
        return self

    def project(self, v: np.ndarray) -> np.ndarray:
        if self.C is None:
            raise RuntimeError("MobiusConnector.fit debe ejecutarse antes de project.")
        return self.C.T @ v

    def lift(self, R_small: np.ndarray) -> np.ndarray:
        """Materializa T completa; O(d²) memoria, reservar para inspección chica."""
        if self.C is None:
            raise RuntimeError("MobiusConnector.fit debe ejecutarse antes de lift.")
        if self.d * self.d > 4_000_000:
            raise MemoryError("T completa sería demasiado grande; usa apply_lift(R_small, v).")
        R_small = np.asarray(R_small, dtype=np.float64)
        if R_small.shape != (self.d_prime, self.d_prime):
            raise ValueError("R_small debe tener forma (d_prime, d_prime).")
        if not np.allclose(R_small.T @ R_small, self._eye_dp, atol=1e-9):
            raise ValueError("R_small debe ser ortogonal.")
        return np.eye(self.d) + self.C @ (R_small - self._eye_dp) @ self.C.T

    def apply_lift(self, R_small: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Aplica T·v en forma de bajo rango, sin crear matrices d×d."""
        if self.C is None:
            raise RuntimeError("MobiusConnector.fit debe ejecutarse antes de apply_lift.")
        R_small = np.asarray(R_small, dtype=np.float64)
        value = np.asarray(v, dtype=np.float64)
        if R_small.shape != (self.d_prime, self.d_prime):
            raise ValueError("R_small debe tener forma (d_prime, d_prime).")
        if not np.all(np.isfinite(R_small)) or not np.allclose(
                R_small.T @ R_small, self._eye_dp, atol=1e-9):
            raise ValueError("R_small debe ser una matriz ortogonal finita.")
        if value.shape != (self.d,) or not np.all(np.isfinite(value)):
            raise ValueError("v debe ser un vector finito de dimensión d.")
        coordinates = self.C.T @ value
        return value + self.C @ ((R_small - self._eye_dp) @ coordinates)

    def rotor_content(self, v: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """
        Rotor content-dependent via Cayley: R = (I − A/2)⁻¹(I + A/2).
        El eje se deriva de p = Cᵀv (CORREGIDO del bug anterior: no se usa
        v como eje del rotor, que lo dejaría fijo).
        """
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (self.d,) or not np.all(np.isfinite(v)):
            raise ValueError("v debe ser un vector finito de dimensión d.")
        if not math.isfinite(scale):
            raise ValueError("scale debe ser finito.")
        p = self.project(v)
        phase_differences = p[:, None] - p[None, :]
        upper = np.triu(scale * 0.5 * np.sin(phase_differences), k=1)
        A = upper - upper.T
        return np.linalg.solve(self._eye_dp - A / 2,
                                self._eye_dp + A / 2)

    def apply(self, v: np.ndarray, R_small: Optional[np.ndarray] = None) -> np.ndarray:
        R = R_small if R_small is not None else self.rotor_content(v)
        return self.apply_lift(R, v)


# ═══════════════════════════════════════════════════════════════════════════
# 4. STATE BRIDGE (Procrustes en z-dominio)
# ═══════════════════════════════════════════════════════════════════════════

class StateBridge:
    """
    Procrustes ortogonal con escala: W = (s_Y/s_X) · U Vᵀ.

    Ajusta un mapa afín escalado con factor ortogonal rectangular. La forma
    cerrada minimiza el error de entrenamiento dentro de esa familia; un
    error bajo en las anclas no garantiza generalización. En el camino de
    dimensiones grandes, `max_anchor_rank=128` limita por defecto el cálculo
    mediante SVD aleatoria; `None` recupera la descomposición completa.
    """
    def __init__(self, max_dense_elements: int = 2_000_000,
                 max_anchor_rank: Optional[int] = 128,
                 n_oversamples: int = 10, n_iter: int = 4):
        if max_dense_elements < 1:
            raise ValueError("max_dense_elements debe ser positivo.")
        if (max_anchor_rank is not None and
                (not isinstance(max_anchor_rank, (int, np.integer)) or
                 isinstance(max_anchor_rank, bool) or max_anchor_rank < 1)):
            raise ValueError("max_anchor_rank debe ser un entero positivo o None.")
        if (not isinstance(n_oversamples, (int, np.integer)) or isinstance(n_oversamples, bool)
                or n_oversamples < 0 or not isinstance(n_iter, (int, np.integer))
                or isinstance(n_iter, bool) or n_iter < 0):
            raise ValueError("n_oversamples y n_iter deben ser enteros no negativos.")
        self.max_dense_elements = int(max_dense_elements)
        self.max_anchor_rank = (int(max_anchor_rank)
                                if max_anchor_rank is not None else None)
        self.n_oversamples, self.n_iter = int(n_oversamples), int(n_iter)
        self.W: Optional[np.ndarray] = None
        self.scale: Optional[float] = None
        self.bias: Optional[np.ndarray] = None
        self.left: Optional[np.ndarray] = None
        self.right: Optional[np.ndarray] = None
        self.source_dim: Optional[int] = None
        self.target_dim: Optional[int] = None
        self.representation = "unfitted"
        self.out_of_anchor = "unfitted"
        self.effective_rank = 0
        self.svd_truncated = False

    def fit(self, X_src: np.ndarray, Y_tgt: np.ndarray) -> 'StateBridge':
        X_src = np.asarray(X_src, dtype=np.float64)
        Y_tgt = np.asarray(Y_tgt, dtype=np.float64)
        if (X_src.ndim != 2 or Y_tgt.ndim != 2 or
                X_src.shape[0] != Y_tgt.shape[0] or X_src.shape[0] == 0):
            raise ValueError("X_src e Y_tgt deben ser matrices con filas pareadas.")
        if not np.all(np.isfinite(X_src)) or not np.all(np.isfinite(Y_tgt)):
            raise ValueError("Las anclas contienen NaN o infinito.")
        x_mean, y_mean = X_src.mean(0), Y_tgt.mean(0)
        Xc, Yc = X_src - x_mean, Y_tgt - y_mean
        denom = float(np.linalg.norm(Xc, 'fro') ** 2)
        self.source_dim, self.target_dim = X_src.shape[1], Y_tgt.shape[1]
        self.svd_truncated = False
        if X_src.shape[1] * Y_tgt.shape[1] <= self.max_dense_elements:
            # El camino denso conserva exactamente el comportamiento previo
            # para espacios pequeños, donde materializar W es razonable.
            U, S, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
            Q = U @ Vt
            self.scale = float(S.sum() / max(denom, 1e-15))
            self.W = self.scale * Q
            self.bias = y_mean - x_mean @ self.W
            self.left = self.right = None
            self.representation = "dense_procrustes"
            self.out_of_anchor = (
                "dense SVD completion; null directions are not identified by anchors")
            self.effective_rank = int(np.count_nonzero(
                S > (np.finfo(float).eps * max(Xc.shape + Yc.shape) * S[0]
                     if len(S) and S[0] > 0 else 0.0)))
            return self

        # En dimensiones grandes no se crea XᵀY (dA×dB). Se descompone cada
        # matriz de anclas con el rango configurado y se hace la SVD restante
        # en el espacio compacto. El puente guarda una isometría parcial de rango
        # observado; componentes fuera del subespacio de anclas se proyectan
        # a cero. Esa limitación es explícita y el rango se reporta.
        def decompose_anchors(matrix: np.ndarray, seed: int):
            available = min(matrix.shape)
            rank_limit = (available if self.max_anchor_rank is None else
                          min(self.max_anchor_rank, available))
            if rank_limit == available:
                return np.linalg.svd(matrix, full_matrices=False)
            self.svd_truncated = True
            return _randomized_svd(
                matrix, rank_limit, n_oversamples=self.n_oversamples,
                n_iter=self.n_iter, seed=seed)

        Ux, sx, Vtx = decompose_anchors(Xc, seed=0)
        Uy, sy, Vty = decompose_anchors(Yc, seed=1)
        rx = self._numerical_rank(sx, Xc.shape)
        ry = self._numerical_rank(sy, Yc.shape)
        if rx == 0 or ry == 0:
            self.left = np.zeros((X_src.shape[1], 0), dtype=np.float64)
            self.right = np.zeros((Y_tgt.shape[1], 0), dtype=np.float64)
            self.scale = 0.0
            self.effective_rank = 0
        else:
            small_cross = (sx[:rx, None] * (Ux[:, :rx].T @ Uy[:, :ry])
                           * sy[None, :ry])
            U_small, singular, Vt_small = np.linalg.svd(
                small_cross, full_matrices=False)
            rank = self._numerical_rank(singular, small_cross.shape)
            self.left = Vtx[:rx].T @ U_small[:, :rank]
            self.right = Vty[:ry].T @ Vt_small[:rank].T
            self.scale = float(singular[:rank].sum() / max(denom, 1e-15))
            self.effective_rank = int(rank)
        self.W = None
        self.bias = y_mean - self.scale * ((x_mean @ self.left) @ self.right.T)
        if self.svd_truncated:
            self.representation = "anchor_randomized_low_rank_procrustes"
            self.out_of_anchor = (
                f"approximate randomized anchor subspaces capped at rank "
                f"{self.max_anchor_rank}; directions outside them map to target mean")
        else:
            self.representation = "anchor_low_rank_procrustes"
            self.out_of_anchor = "directions outside the observed anchor span map to target mean"
        return self

    @staticmethod
    def _numerical_rank(singular: np.ndarray, shape: Tuple[int, ...]) -> int:
        if len(singular) == 0 or singular[0] <= 0:
            return 0
        threshold = np.finfo(float).eps * max(shape) * singular[0]
        return int(np.count_nonzero(singular > threshold))

    def align(self, v: np.ndarray) -> np.ndarray:
        if self.bias is None or self.source_dim is None:
            raise RuntimeError("StateBridge.fit debe ejecutarse antes de align.")
        value = np.asarray(v, dtype=np.float64)
        if (value.ndim not in (1, 2) or value.shape[-1] != self.source_dim
                or not np.all(np.isfinite(value))):
            raise ValueError("La última dimensión de entrada no coincide con el bridge.")
        if self.W is not None:
            return value @ self.W + self.bias
        if self.left is None or self.right is None or self.scale is None:
            raise RuntimeError("El bridge de bajo rango no quedó ajustado.")
        return self.scale * ((value @ self.left) @ self.right.T) + self.bias


class MatryoshkaBridge:
    """Copia el prefijo común cuando el encoder garantiza dimensiones anidadas.

    No alinea encoders distintos. Si el origen es más ancho que el destino,
    truncar pierde componentes y el reporte deja constancia de esa pérdida.
    """
    def __init__(self, source_dim: int, target_dim: int):
        if source_dim < 1 or target_dim < 1:
            raise ValueError("Las dimensiones Matryoshka deben ser positivas.")
        self.source_dim, self.target_dim = int(source_dim), int(target_dim)

    def align(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (self.source_dim,) or not np.all(np.isfinite(value)):
            raise ValueError("El vector no coincide con la dimensión Matryoshka de origen.")
        result = np.zeros(self.target_dim, dtype=np.float64)
        shared = min(self.source_dim, self.target_dim)
        result[:shared] = value[:shared]
        return result

    @property
    def lost_dimensions(self) -> int:
        return max(0, self.source_dim - self.target_dim)

    @property
    def out_of_anchor(self) -> str:
        return "copies the prefix under the declared nested-coordinate assumption"

    @property
    def representation(self) -> str:
        return "nested_prefix_assumption"

    @property
    def effective_rank(self) -> int:
        return min(self.source_dim, self.target_dim)


# ═══════════════════════════════════════════════════════════════════════════
# 5. LOG STATE BRIDGE (Procrustes en log-dominio — maneja T_A ≠ T_B)
# ═══════════════════════════════════════════════════════════════════════════

def _log_transform(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """log|x|; el signo se transporta en un canal separado."""
    return np.log(np.abs(M) + eps)


def _inv_log_transform(M_log: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(M_log, -700.0, 700.0))


class _IdentitySignBridge:
    """Puente sin ajuste cuando origen y destino tienen exactamente los mismos signos."""
    representation = "identity_sign"
    effective_rank = 0

    def align(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64).copy()


class LogStateBridge:
    """
    Procrustes en el dominio s = log|z|.

    Heurística útil cuando la relación observada se parece a una ley de
    potencia firmada. Que T_A y T_B difieran no demuestra por sí solo que ese
    modelo sea apropiado; `build_bridge` compara su error reservado.
    """
    def __init__(self, max_anchor_rank: Optional[int] = 128,
                 n_oversamples: int = 10, n_iter: int = 4):
        self.W: Optional[np.ndarray] = None
        self.shift: Optional[np.ndarray] = None
        self.scale_ratio: Optional[float] = None
        self.magnitude_bridge: Optional[StateBridge] = None
        self.sign_bridge: Optional[object] = None
        self.eps = 1e-12
        self.max_anchor_rank = max_anchor_rank
        self.n_oversamples, self.n_iter = n_oversamples, n_iter

    def fit(self, X_src: np.ndarray, Y_tgt: np.ndarray) -> 'LogStateBridge':
        Xl = _log_transform(np.asarray(X_src, dtype=np.float64), self.eps)
        Yl = _log_transform(np.asarray(Y_tgt, dtype=np.float64), self.eps)
        bridge = StateBridge(max_anchor_rank=self.max_anchor_rank,
                             n_oversamples=self.n_oversamples,
                             n_iter=self.n_iter).fit(Xl, Yl)
        self.magnitude_bridge = bridge
        self.W, self.shift = bridge.W, bridge.bias
        self.scale_ratio = bridge.scale
        # El logaritmo sólo modela magnitudes. Se ajusta por separado la señal
        # de signo; para la ley de potencia elemento a elemento, el signo se
        # conserva exactamente.
        source_signs = np.sign(np.asarray(X_src, dtype=np.float64))
        target_signs = np.sign(np.asarray(Y_tgt, dtype=np.float64))
        if source_signs.shape == target_signs.shape and np.array_equal(
                source_signs, target_signs):
            self.sign_bridge = _IdentitySignBridge()
        else:
            self.sign_bridge = StateBridge(
                max_anchor_rank=self.max_anchor_rank,
                n_oversamples=self.n_oversamples,
                n_iter=self.n_iter).fit(source_signs, target_signs)
        return self

    def align(self, v: np.ndarray) -> np.ndarray:
        if self.magnitude_bridge is None:
            raise RuntimeError("LogStateBridge.fit debe ejecutarse antes de align.")
        if self.sign_bridge is None:
            raise RuntimeError("LogStateBridge no tiene puente de signos ajustado.")
        value = np.asarray(v, dtype=np.float64)
        v_log = _log_transform(value, self.eps)
        magnitudes = _inv_log_transform(self.magnitude_bridge.align(v_log))
        sign_prediction = np.sign(self.sign_bridge.align(np.sign(value)))
        return sign_prediction * magnitudes

    @property
    def representation(self) -> str:
        mag = (self.magnitude_bridge.representation if self.magnitude_bridge is not None
               else "unfitted")
        sign = getattr(self.sign_bridge, "representation", "unfitted")
        return f"log-magnitude:{mag};sign:{sign}"

    @property
    def out_of_anchor(self) -> str:
        return (self.magnitude_bridge.out_of_anchor if self.magnitude_bridge is not None
                else "unfitted")

    @property
    def effective_rank(self) -> int:
        magnitude_rank = (self.magnitude_bridge.effective_rank
                          if self.magnitude_bridge is not None else 0)
        sign_rank = getattr(self.sign_bridge, "effective_rank", 0)
        return int(max(magnitude_rank, sign_rank))

    @property
    def svd_truncated(self) -> bool:
        return bool(
            getattr(self.magnitude_bridge, "svd_truncated", False) or
            getattr(self.sign_bridge, "svd_truncated", False))


# ═══════════════════════════════════════════════════════════════════════════
# 6. ADAPTADOR NO-LINEAL (Nivel 2 — Kernel Ridge cerrado, sin torch)
# ═══════════════════════════════════════════════════════════════════════════

class KernelRidgeAdapter:
    """
    Adaptador no-lineal en forma cerrada: Kernel Ridge Regression.

    Modelo: v_B ≈ K(v_A, X_train) · (K + λI)⁻¹ · Y_train
    Kernel: RBF (Gaussiano) sobre similitud coseno.

    Resuelve Kernel Ridge para un kernel RBF de similitud coseno. El ajuste
    minimiza el objetivo regularizado en entrenamiento; la calidad fuera de
    muestra depende de los datos, gamma y regularización.
    """
    def __init__(self, kernel: str = "rbf", gamma: float = 1.0, lam: float = 1e-3):
        self.kernel, self.gamma, self.lam = kernel, gamma, lam
        self.X_train: Optional[np.ndarray] = None
        self.alpha_coef: Optional[np.ndarray] = None
        self.representation = "kernel_ridge_cosine_rbf"
        self.out_of_anchor = "kernel-weighted interpolation from training anchors"

    def _K(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Kernel RBF sobre similitud coseno."""
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        cos = An @ Bn.T
        return np.exp(-self.gamma * (1.0 - cos))

    def fit(self, X_src: np.ndarray, Y_tgt: np.ndarray) -> 'KernelRidgeAdapter':
        X_src = np.asarray(X_src, dtype=np.float64)
        Y_tgt = np.asarray(Y_tgt, dtype=np.float64)
        if (X_src.ndim != 2 or Y_tgt.ndim != 2 or
                len(X_src) != len(Y_tgt) or len(X_src) == 0):
            raise ValueError("KernelRidge requiere anclas pareadas no vacías.")
        if self.gamma <= 0 or self.lam <= 0:
            raise ValueError("gamma y lam deben ser positivos.")
        self.X_train = X_src.copy()
        K = self._K(X_src, X_src)
        self.alpha_coef = np.linalg.solve(K + self.lam * np.eye(len(K)), Y_tgt)
        return self

    def align(self, v: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.alpha_coef is None:
            raise RuntimeError("KernelRidgeAdapter.fit debe ejecutarse antes de align.")
        v = np.asarray(v)
        if v.ndim == 2:
            if v.shape[1] != self.X_train.shape[1]:
                raise ValueError("La dimensión de entrada no coincide con las anclas.")
            return self._K(v, self.X_train) @ self.alpha_coef
        if v.ndim != 1:
            raise ValueError("align espera un vector 1D o un lote 2D.")
        k = self._K(v[None, :], self.X_train)
        return (k @ self.alpha_coef).flatten()

    @property
    def effective_rank(self) -> int:
        return int(len(self.X_train)) if self.X_train is not None else 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. BRIDGE FACTORY (regla de decisión Nivel 0/1/2)
# ═══════════════════════════════════════════════════════════════════════════

def build_bridge(X_src: np.ndarray, Y_tgt: np.ndarray,
                  T_src: float, T_tgt: float,
                  err_threshold: float = 0.10,
                  seed: int = 0,
                  nested_coordinates: bool = False,
                  max_bridge_rank: Optional[int] = 128,
                  max_kernel_anchors: Optional[int] = 2000,
                  svd_oversamples: int = 10,
                  svd_power_iterations: int = 4) -> Tuple[Callable, Dict]:
    """
    Regla secuencial (NO alternativa):
      1. Si T_src == T_tgt → StateBridge (z-dominio)
      2. Si T_src != T_tgt → LogStateBridge (log-dominio, warp de frecuencia)
      3. Medir err_rel; si > err_threshold → KernelRidgeAdapter (Nivel 2)

    `max_bridge_rank` limita la aproximación SVD de los bridges grandes;
    `max_kernel_anchors` limita tanto la validación como el ajuste final de KRR.
    `svd_oversamples` y `svd_power_iterations` permiten ajustar el costo y la
    precisión del rango aleatorio.

    DEVUELVE (align_fn, meta) con meta describiendo qué nivel se eligió.
    """
    X_src = np.asarray(X_src, dtype=np.float64)
    Y_tgt = np.asarray(Y_tgt, dtype=np.float64)
    if (X_src.ndim != 2 or Y_tgt.ndim != 2 or
            len(X_src) != len(Y_tgt) or len(X_src) < 2):
        raise ValueError("El bridge requiere al menos dos pares de anclas.")
    if err_threshold < 0:
        raise ValueError("err_threshold debe ser no negativo.")
    if (max_bridge_rank is not None and
            (not isinstance(max_bridge_rank, (int, np.integer)) or
             isinstance(max_bridge_rank, bool) or max_bridge_rank < 1)):
        raise ValueError("max_bridge_rank debe ser un entero positivo o None.")
    if (max_kernel_anchors is not None and
            (not isinstance(max_kernel_anchors, (int, np.integer)) or
             isinstance(max_kernel_anchors, bool) or max_kernel_anchors < 4)):
        raise ValueError("max_kernel_anchors debe ser un entero >= 4 o None.")
    if (not isinstance(svd_oversamples, (int, np.integer)) or
            isinstance(svd_oversamples, bool) or svd_oversamples < 0 or
            not isinstance(svd_power_iterations, (int, np.integer)) or
            isinstance(svd_power_iterations, bool) or svd_power_iterations < 0):
        raise ValueError("svd_oversamples y svd_power_iterations deben ser enteros no negativos.")

    def make_level1(domain: str):
        if domain == "log":
            return LogStateBridge(max_anchor_rank=max_bridge_rank,
                                  n_oversamples=svd_oversamples,
                                  n_iter=svd_power_iterations)
        return StateBridge(max_anchor_rank=max_bridge_rank,
                           n_oversamples=svd_oversamples,
                           n_iter=svd_power_iterations)

    if nested_coordinates:
        # Se activa sólo bajo una garantía externa del encoder. Las muestras
        # no prueban por sí mismas que los ejes sean prefijos anidados.
        bridge = MatryoshkaBridge(X_src.shape[1], Y_tgt.shape[1])
        return bridge.align, {"level": 0, "domain": "nested-prefix",
                              "validation_error_nivel1": None,
                              "validation_error_nivel2": None,
                              "validation_count": 0,
                              "assumption_verified": False,
                              "lost_dimensions": bridge.lost_dimensions,
                              "bridge_representation": bridge.representation,
                              "bridge_rank": bridge.effective_rank,
                              "bridge_out_of_anchor": bridge.out_of_anchor,
                              "bridge": bridge}

    # Reserva datos para elegir el método; nunca compara métodos sobre las
    # mismas filas que ajustaron sus parámetros, salvo corpus demasiado chico.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_src))
    n_val = max(1, int(round(0.25 * len(order)))) if len(order) >= 5 else 0
    val_idx, train_idx = order[:n_val], order[n_val:]
    in_sample = n_val == 0
    if in_sample:
        train_idx = order
    X_train, Y_train = X_src[train_idx], Y_tgt[train_idx]
    X_val = X_src[val_idx] if n_val else X_train
    Y_val = Y_tgt[val_idx] if n_val else Y_train

    def rel_error(model: object, X: np.ndarray, Y: np.ndarray) -> float:
        # Los bridges matriciales aceptan un lote; evita una llamada Python
        # por ancla durante la selección sin cambiar el error agregado.
        pred = np.asarray(model.align(np.asarray(X, dtype=np.float64)))
        return float(np.linalg.norm(pred - Y) /
                     max(float(np.linalg.norm(Y)), 1e-15))

    prefer_log = abs(T_src - T_tgt) >= 1e-12
    if in_sample:
        # Con menos de cinco pares no hay holdout. Elegir entre dos métodos
        # por su error de entrenamiento sería engañoso y duplicaría SVDs.
        domain = "log" if prefer_log else "z"
        selected = make_level1(domain).fit(X_src, Y_tgt)
        error1 = rel_error(selected, X_src, Y_tgt)
        meta = {"level": 1, "domain": domain,
                "validation_error_nivel1": error1,
                "validation_error_nivel2": None,
                "validation_count": 0, "selection_in_sample": True,
                "bridge_representation": getattr(selected, "representation", "unknown"),
                "bridge_rank": getattr(selected, "effective_rank", None),
                "bridge_svd_truncated": getattr(selected, "svd_truncated", False),
                "max_bridge_rank": max_bridge_rank,
                "svd_oversamples": svd_oversamples,
                "svd_power_iterations": svd_power_iterations,
                "bridge_out_of_anchor": getattr(selected, "out_of_anchor", "not specified"),
                "bridge": selected}
        return selected.align, meta

    candidates = []
    for domain in ("z", "log"):
        model = make_level1(domain).fit(X_train, Y_train)
        score = rel_error(model, X_val, Y_val)
        candidates.append((score, 0 if (domain == "log") == prefer_log else 1,
                           domain, model))
    candidates.sort(key=lambda item: (item[0], item[1]))
    error1, _, domain, selected = candidates[0]
    level = 1
    error2 = None

    # Nivel 2 sólo se considera si hay validación suficiente. El umbral y el
    # requisito de mejora son decisiones de ingeniería, no teoremas.
    kernel_ridge_validation_anchor_count = None
    kernel_ridge_anchor_count = None
    kernel_ridge_subsampled = False
    if error1 > err_threshold and n_val and len(train_idx) >= 4:
        kernel_train_idx = train_idx
        if (max_kernel_anchors is not None and
                len(kernel_train_idx) > max_kernel_anchors):
            kernel_train_idx = np.sort(rng.choice(
                kernel_train_idx, size=max_kernel_anchors, replace=False))
        kernel_ridge_validation_anchor_count = int(len(kernel_train_idx))
        kr = KernelRidgeAdapter().fit(X_src[kernel_train_idx], Y_tgt[kernel_train_idx])
        error2 = rel_error(kr, X_val, Y_val)
        if error2 < error1 * 0.8:
            selected, domain, level = kr, "cosine-rbf", 2

    # Reajusta el método elegido con todas las anclas después de seleccionar.
    if level == 1:
        selected = make_level1(domain).fit(X_src, Y_tgt)
    else:
        final_kernel_idx = np.arange(len(X_src))
        if (max_kernel_anchors is not None and
                len(final_kernel_idx) > max_kernel_anchors):
            final_kernel_idx = np.sort(rng.choice(
                final_kernel_idx, size=max_kernel_anchors, replace=False))
        kernel_ridge_anchor_count = int(len(final_kernel_idx))
        kernel_ridge_subsampled = kernel_ridge_anchor_count < len(X_src)
        selected = KernelRidgeAdapter().fit(
            X_src[final_kernel_idx], Y_tgt[final_kernel_idx])
    meta = {"level": level, "domain": domain,
            "validation_error_nivel1": error1,
            "validation_error_nivel2": error2,
            "validation_count": int(n_val), "selection_in_sample": in_sample,
            "bridge_representation": getattr(selected, "representation", "unknown"),
            "bridge_rank": getattr(selected, "effective_rank", None),
            "bridge_svd_truncated": getattr(selected, "svd_truncated", False),
            "max_bridge_rank": max_bridge_rank,
            "svd_oversamples": svd_oversamples,
            "svd_power_iterations": svd_power_iterations,
            "kernel_ridge_validation_anchor_count": kernel_ridge_validation_anchor_count,
            "kernel_ridge_anchor_count": kernel_ridge_anchor_count,
            "kernel_ridge_anchor_limit": max_kernel_anchors,
            "kernel_ridge_subsampled": kernel_ridge_subsampled,
            "bridge_out_of_anchor": getattr(selected, "out_of_anchor", "not specified"),
            "bridge": selected}
    return selected.align, meta


# ═══════════════════════════════════════════════════════════════════════════
# 8. TTL REALIGNER (polar periódico)
# ═══════════════════════════════════════════════════════════════════════════

class TTLRealigner:
    """
    Re-ortogonaliza la matriz acumulada cada `ttl` interacciones usando polar.

    La periodicidad es una decisión operativa. No se deriva de la fórmula
    z-Laplace ni implica un ciclo físico. Cada composición acumula redondeo;
    este componente mide la deriva y proyecta periódicamente al factor polar.
    """
    def __init__(self, ttl: int = 2048, tol: float = 1e-10):
        if ttl < 1 or tol <= 0:
            raise ValueError("ttl y tol deben ser positivos.")
        self.ttl, self.tol = ttl, tol
        self.count = 0
        self.drift_log: List[float] = []

    def accumulate(self, R_accum: np.ndarray, R_new: np.ndarray) -> np.ndarray:
        R_accum = np.asarray(R_accum, dtype=np.float64)
        R_new = np.asarray(R_new, dtype=np.float64)
        if (R_accum.ndim != 2 or R_accum.shape[0] != R_accum.shape[1]
                or R_new.shape != R_accum.shape):
            raise ValueError("Los rotores acumulado y nuevo deben ser cuadrados y compatibles.")
        if not np.all(np.isfinite(R_accum)) or not np.all(np.isfinite(R_new)):
            raise ValueError("Los rotores contienen NaN o infinito.")
        R_accum = R_new @ R_accum
        self.count += 1
        drift = ortho_drift(R_accum)
        self.drift_log.append(drift)
        if self.count % self.ttl == 0 or drift > self.tol:
            R_accum = polar_orthogonal(R_accum, special=True)
            self.drift_log.append(-1.0)   # marcador de realineación
        return R_accum

    def stats(self) -> Dict:
        reals = sum(1 for d in self.drift_log if d == -1.0)
        drifts = [d for d in self.drift_log if d >= 0]
        return {"interactions": self.count, "realignments": reals,
                "max_drift": max(drifts, default=0.0),
                "final_drift": drifts[-1] if drifts else 0.0}

    def reset(self) -> None:
        self.count = 0
        self.drift_log.clear()


# ═══════════════════════════════════════════════════════════════════════════
# 9. FRACTAL AVERAGER (promediado sobre 6 fases kπ/3)
# ═══════════════════════════════════════════════════════════════════════════

class FractalAverager:
    """
    Genera seis réplicas de fase, las devuelve al mismo marco y promedia.
    Sin ruido explícito el promedio es exactamente la identidad y se devuelve
    sin construir operadores. `Agent` sólo activa el promedio con ruido cuando
    `fractal_noise_std` es mayor que cero.
    Con ruido aditivo independiente y centrado, la varianza del ruido medio
    es 1/6 de la varianza de una muestra. Para ruido correlacionado o sesgado
    esa reducción no está garantizada. El promedio no es una transformación
    de Möbius ni conserva necesariamente la norma si hay ruido.
    """
    def __init__(self, connector: MobiusConnector):
        self.conn = connector
        phases = []
        if self.conn.d_prime >= 2:
            for k in range(6):
                phase = self._phase_rotation(k * np.pi / 3)
                phase.setflags(write=False)
                phases.append(phase)
        self._phases = tuple(phases)
        self._phase_transposes = tuple(phase.T for phase in self._phases)

    def _phase_rotation(self, theta: float) -> np.ndarray:
        R = np.eye(self.conn.d_prime)
        c, s = np.cos(theta), np.sin(theta)
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
        return R

    def apply(self, v: np.ndarray, noise_std: float = 0.0,
              seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (self.conn.d,) or not np.all(np.isfinite(v)):
            raise ValueError("v debe ser un vector finito compatible con el conector.")
        if noise_std < 0 or not np.isfinite(noise_std):
            raise ValueError("noise_std debe ser finito y no negativo.")
        # Sin ruido, cada fase y su inversa se cancelan exactamente. No se
        # materializan seis operadores idénticos al resultado identidad.
        if noise_std == 0.0:
            return v.copy(), np.zeros_like(v)
        if not self._phases:
            raise ValueError("El promedio por seis fases requiere d_prime >= 2.")
        rng = np.random.default_rng(seed)
        aligned = []
        for phase_transpose in self._phase_transposes:
            noise = rng.normal(0.0, noise_std, size=v.shape)
            # Tᵀ(Tv + n) = v + Tᵀn porque el levantamiento T es ortogonal.
            aligned.append(v + self.conn.apply_lift(phase_transpose, noise))
        values = np.stack(aligned, axis=0)
        return values.mean(axis=0), values.std(axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Z-LAPLACE VERIFIER (consistencia de T declarado vs. calculado)
# ═══════════════════════════════════════════════════════════════════════════

def verify_z_laplacian(T_declared: float, n_tilde: float, tol: float = 1e-12) -> Dict:
    """
    Verifica que T_declared == 1/7 + 4/(7·ñ).
    Si no coincide, el agente opera en una escala DISTINTA a la declarada,
    y la alineación lineal fallará por causa del z-Laplaciano, no del bridge.

    IDENTIDAD: para ñ=10, T = 1/7 + 4/70 = 1/5. ttl_natural = 5.
    """
    if not (math.isfinite(T_declared) and T_declared > 0 and
            math.isfinite(n_tilde) and n_tilde > 0):
        raise ValueError("T_declared y n_tilde deben ser positivos y finitos.")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol debe ser positivo y finito.")
    T_expected = 1.0 / 7.0 + 4.0 / (7.0 * n_tilde)
    return {"T_declared": T_declared, "T_expected": T_expected,
            "consistent": abs(T_declared - T_expected) < tol,
            "ttl_natural": int(round(1.0 / max(T_declared, 1e-12)))}


# ═══════════════════════════════════════════════════════════════════════════
# 11. DECODE INTERFACE (interfaz aparte, preserva original)
# ═══════════════════════════════════════════════════════════════════════════

class DecodeInterface:
    """
    Interfaz de decodificación SEPARADA del flujo interno.
    El multivector original NUNCA se modifica. Se decodifica una copia.

    PARALELO con bases de datos vectoriales: consultar no modifica el índice.
    """
    def __init__(self, decoder_fn: Optional[Callable[[np.ndarray], str]] = None):
        self.decoder_fn = decoder_fn or (lambda v: f"<latent dim={len(v)}>")

    def decode(self, v_latent: np.ndarray) -> Dict:
        original = np.asarray(v_latent, dtype=np.float64)
        if original.ndim != 1 or not np.all(np.isfinite(original)):
            raise ValueError("DecodeInterface requiere un vector finito.")
        v_copy = original.copy()
        h = hashlib.sha256(original.tobytes()).hexdigest()[:16]
        text = self.decoder_fn(v_copy)
        return {"text": text, "hash_original": h,
                "norm_original": float(np.linalg.norm(original))}


class Vec2TextDecoder:
    """Adaptador opcional y perezoso de Vec2Text.

    Vec2Text requiere el espacio nativo del embedder para el que se entrenó su
    corrector. Por eso esta clase exige `to_native_space`; no asume que un
    vector pasado por bridge/rotor sea invertible ni compatible con Ada.
    La dependencia pesada sólo se importa al primer decode.
    """
    def __init__(self, model_name: str = "text-embedding-ada-002",
                 num_steps: int = 20, beam_width: int = 0,
                 to_native_space: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        if num_steps < 0 or beam_width < 0:
            raise ValueError("num_steps y beam_width deben ser no negativos.")
        self.model_name = model_name
        self.num_steps = int(num_steps)
        self.beam_width = int(beam_width)
        self.to_native_space = to_native_space
        self._corrector = None

    @property
    def corrector(self):
        if self._corrector is None:
            try:
                import vec2text
            except ImportError as exc:
                raise RuntimeError("Instala vec2text para habilitar esta hipótesis de texto.") from exc
            self._corrector = vec2text.load_pretrained_corrector(self.model_name)
        return self._corrector

    def __call__(self, v_latent: np.ndarray) -> str:
        if self.to_native_space is None:
            raise RuntimeError("Vec2Text necesita una transformación inversa al espacio nativo.")
        import vec2text
        native = np.asarray(self.to_native_space(np.asarray(v_latent, dtype=np.float64).copy()),
                            dtype=np.float32)
        if native.ndim == 1:
            native = native[None, :]
        if native.ndim != 2 or native.shape[0] != 1 or not np.all(np.isfinite(native)):
            raise ValueError("to_native_space debe devolver un vector o batch de un vector finito.")
        result = vec2text.invert_embeddings(
            embeddings=native,
            corrector=self.corrector,
            num_steps=self.num_steps,
            sequence_beam_width=self.beam_width,
        )
        return result[0] if isinstance(result, (list, tuple)) else str(result)

    def decode(self, v_latent: np.ndarray) -> Dict:
        """Interfaz compatible con Protocol, con hash del estado consultado."""
        original = np.asarray(v_latent, dtype=np.float64)
        if original.ndim != 1 or not np.all(np.isfinite(original)):
            raise ValueError("Vec2TextDecoder requiere un vector finito.")
        return {"text": self(original.copy()),
                "hash_original": hashlib.sha256(original.tobytes()).hexdigest()[:16],
                "norm_original": float(np.linalg.norm(original))}


class RAGDecodeInterface:
    """Recupera texto/cita desde un índice vectorial ya alineado al espacio B.

    El índice debe haberse creado con el mismo encoder y transformaciones que
    producen `v_latent`. Similaridad alta es un umbral operativo, no una prueba
    de identidad textual. Vec2Text, si se proporciona, sólo aporta hipótesis.
    """
    def __init__(self, corpus_embeddings: Optional[np.ndarray] = None,
                 corpus_texts: Optional[List[str]] = None,
                 vec2text_decoder: Optional[Callable[[np.ndarray], str]] = None,
                 top_k: int = 5, confidence_threshold: float = 0.95,
                 corpus_ids: Optional[List[str]] = None):
        if top_k < 1 or not -1.0 <= confidence_threshold <= 1.0:
            raise ValueError("top_k debe ser positivo y el umbral debe estar entre -1 y 1.")
        self.vec2text = vec2text_decoder
        self.top_k = int(top_k)
        self.confidence_threshold = float(confidence_threshold)
        self.corpus_emb: Optional[np.ndarray] = None
        self.corpus_texts: List[str] = []
        self.corpus_ids: List[str] = []
        if corpus_embeddings is not None or corpus_texts is not None:
            if corpus_embeddings is None or corpus_texts is None:
                raise ValueError("Se necesitan juntos corpus_embeddings y corpus_texts.")
            self.fit_index(corpus_embeddings, corpus_texts, corpus_ids)

    def fit_index(self, corpus_embeddings: np.ndarray, corpus_texts: List[str],
                  corpus_ids: Optional[List[str]] = None) -> "RAGDecodeInterface":
        embeddings = np.asarray(corpus_embeddings, dtype=np.float64)
        if (embeddings.ndim != 2 or len(embeddings) == 0 or
                len(embeddings) != len(corpus_texts) or not np.all(np.isfinite(embeddings))):
            raise ValueError("El índice requiere embeddings finitos y un texto por fila.")
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("El índice RAG no admite vectores de norma cero.")
        if corpus_ids is not None and len(corpus_ids) != len(corpus_texts):
            raise ValueError("Debe haber un id/referencia por cada texto.")
        self.corpus_emb = embeddings / norms[:, None]
        self.corpus_texts = [str(text) for text in corpus_texts]
        self.corpus_ids = ([str(value) for value in corpus_ids] if corpus_ids is not None
                           else [str(index) for index in range(len(corpus_texts))])
        self.corpus_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest()
                              for text in self.corpus_texts]
        return self

    def decode(self, v_latent: np.ndarray) -> Dict:
        if self.corpus_emb is None:
            raise RuntimeError("RAGDecodeInterface.fit_index debe ejecutarse antes de decode.")
        original = np.asarray(v_latent, dtype=np.float64)
        if (original.ndim != 1 or original.size != self.corpus_emb.shape[1]
                or not np.all(np.isfinite(original))):
            raise ValueError("La consulta debe ser un vector finito de la dimensión indexada.")
        norm = float(np.linalg.norm(original))
        if norm <= 1e-12:
            raise ValueError("La consulta RAG tiene norma cero.")
        digest = hashlib.sha256(original.tobytes()).hexdigest()[:16]
        hypothesis, hypothesis_error = None, None
        if self.vec2text is not None:
            try:
                hypothesis = self.vec2text(original.copy())
            except Exception as exc:
                hypothesis_error = f"{type(exc).__name__}: {exc}"

        scores = self.corpus_emb @ (original / norm)
        indices = np.argsort(scores)[::-1][:min(self.top_k, len(scores))]
        candidates = [{"id": self.corpus_ids[int(i)],
                       "text": self.corpus_texts[int(i)],
                       "source_hash": self.corpus_hashes[int(i)],
                       "score": float(scores[i])} for i in indices]
        if hypothesis:
            for item in candidates:
                item["lexical_score"] = difflib.SequenceMatcher(
                    None, str(hypothesis), item["text"]).ratio()
            best = max(candidates,
                       key=lambda item: (0.65 * item["score"] +
                                         0.35 * item["lexical_score"]))
        else:
            best = candidates[0]
        accepted = best["score"] >= self.confidence_threshold
        final_text = best["text"] if accepted else (str(hypothesis) if hypothesis else None)
        return {"text": final_text,
                "candidate_text": best["text"],
                "reference": best["id"],
                "source_hash": best["source_hash"],
                "score": best["score"],
                "accepted_from_corpus": bool(accepted),
                "hypothesis": hypothesis,
                "hypothesis_error": hypothesis_error,
                "candidates": candidates,
                "hash_original": digest,
                "norm_original": norm}


class SecureRAGIndex:
    """Verifica IDs, textos y vectores del índice antes de confiar en RAG.

    La firma acredita que el contenido coincide con el índice firmado por quien
    posee la clave. No prueba que la fuente original sea verdadera ni evita
    contaminación ocurrida antes de firmar.
    """
    def __init__(self, rag: RAGDecodeInterface, signing_key: bytes):
        if not isinstance(signing_key, bytes) or len(signing_key) < 32:
            raise ValueError("signing_key debe contener al menos 32 bytes.")
        if not isinstance(rag, RAGDecodeInterface) or rag.corpus_emb is None:
            raise ValueError("SecureRAGIndex requiere un RAG ya indexado.")
        if (rag.corpus_emb.ndim != 2 or len(rag.corpus_emb) == 0 or
                len(rag.corpus_ids) != len(rag.corpus_emb) or
                len(rag.corpus_texts) != len(rag.corpus_emb)):
            raise ValueError("El índice RAG cambió de forma antes de firmarlo.")
        if len(set(rag.corpus_ids)) != len(rag.corpus_ids):
            raise ValueError("El índice seguro requiere IDs de corpus únicos.")
        self.rag = rag
        self._key = _hkdf_sha256(signing_key, b"a2a-mobius/v1/secure-rag")
        self._ids = tuple(rag.corpus_ids)
        self._texts = tuple(rag.corpus_texts)
        self._top_k = int(rag.top_k)
        self._confidence_threshold = float(rag.confidence_threshold)
        self._vec2text = rag.vec2text
        self._embeddings = np.ascontiguousarray(rag.corpus_emb, dtype="<f8").copy()
        self._embeddings.setflags(write=False)
        self._records: Dict[str, Tuple[str, str]] = {}
        self._record_indexes: Dict[str, int] = {}
        for index, (record_id, text) in enumerate(zip(self._ids, self._texts)):
            embedding_hash = hashlib.sha256(self._embeddings[index].tobytes()).hexdigest()
            record = {"id": record_id, "text": text, "embedding_sha256": embedding_hash}
            tag = hmac.new(self._key, b"A2A-MOBIUS-RAG-v1\0" + _canonical_json_bytes(record),
                           hashlib.sha256).hexdigest()
            self._records[record_id] = (text, tag)
            self._record_indexes[record_id] = index
        # El wrapper sirve una instantánea propia, inmutable y verificada. La
        # verificación completa queda fuera del hot path y puede repetirse con verify().
        self._verified = self._verify_index()
        if not self._verified:
            raise RuntimeError("SecureRAGIndex: el índice falló la verificación al construirse.")

    def _verify_index(self) -> bool:
        """Auditoría completa O(N); se usa al crear o invocar verify()."""
        rag = self.rag
        if (rag.corpus_emb is None or rag.corpus_emb.shape != self._embeddings.shape or
                tuple(rag.corpus_ids) != self._ids or tuple(rag.corpus_texts) != self._texts):
            return False
        current = np.ascontiguousarray(rag.corpus_emb, dtype="<f8")
        if not np.all(np.isfinite(current)) or not np.array_equal(current, self._embeddings):
            return False
        return all(self._verify_record(index) for index in range(len(self._ids)))

    def _verify_record(self, index: int) -> bool:
        """Verifica un registro de la instantánea con trabajo constante por fila."""
        if not 0 <= index < len(self._ids):
            return False
        record_id, text = self._ids[index], self._texts[index]
        stored = self._records.get(record_id)
        if stored is None or stored[0] != text:
            return False
        row = np.ascontiguousarray(self._embeddings[index], dtype="<f8")
        if not np.all(np.isfinite(row)):
            return False
        record = {"id": record_id, "text": text,
                  "embedding_sha256": hashlib.sha256(row.tobytes()).hexdigest()}
        expected = hmac.new(self._key, b"A2A-MOBIUS-RAG-v1\0" + _canonical_json_bytes(record),
                            hashlib.sha256).hexdigest()
        return hmac.compare_digest(stored[1], expected)

    def verify(self) -> bool:
        """Revalida explícitamente el índice fuente completo, fuera de cada query."""
        self._verified = self._verify_index()
        return self._verified

    def decode(self, v_latent: np.ndarray) -> Dict:
        if not self._verified:
            raise RuntimeError("SecureRAGIndex: índice no verificado; ejecuta verify().")
        query = np.asarray(v_latent, dtype=np.float64)
        if (query.ndim != 1 or query.shape[0] != self._embeddings.shape[1] or
                not np.all(np.isfinite(query))):
            raise ValueError("La consulta debe ser un vector finito de la dimensión indexada.")
        norm = float(np.linalg.norm(query))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("La consulta RAG tiene norma cero.")

        # Se calcula la similitud una sola vez sobre la instantánea firmada.
        # La búsqueda exacta sigue siendo O(N·d); la autenticidad de filas
        # candidatas pasa de O(N) serializaciones/HMAC a O(k).
        scores = self._embeddings @ (query / norm)
        count = min(self._top_k, len(scores))
        if count < len(scores):
            indexes = np.argpartition(scores, len(scores) - count)[-count:]
        else:
            indexes = np.arange(len(scores))
        indexes = indexes[np.argsort(scores[indexes])[::-1]]
        candidates = []
        for index in indexes:
            index = int(index)
            if not self._verify_record(index):
                self._verified = False
                raise RuntimeError("SecureRAGIndex: falló la firma de una fila candidata.")
            record_id, text = self._ids[index], self._texts[index]
            candidates.append({
                "id": record_id,
                "text": text,
                "source_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "score": float(scores[index]),
            })

        hypothesis, hypothesis_error = None, None
        if self._vec2text is not None:
            try:
                hypothesis = self._vec2text(query.copy())
            except Exception as exc:
                hypothesis_error = f"{type(exc).__name__}: {exc}"
        best = candidates[0]
        accepted = best["score"] >= self._confidence_threshold
        return {
            "text": best["text"] if accepted else None,
            "candidate_text": best["text"],
            "reference": best["id"],
            "source_hash": best["source_hash"],
            "score": best["score"],
            "accepted_from_corpus": bool(accepted),
            "corpus_signature_verified": True,
            "hypothesis": hypothesis,
            "hypothesis_error": hypothesis_error,
            "candidates": candidates,
            "hash_original": hashlib.sha256(query.tobytes()).hexdigest()[:16],
            "norm_original": norm,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 12. TASK / TASK TURN (versionado de estado entre turnos)
# ═══════════════════════════════════════════════════════════════════════════

class TaskState(Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskTurn:
    turn: int
    timestamp: str
    latent: np.ndarray
    hash_latent: str
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = np.asarray(self.latent, dtype=np.float64).copy()
        if state.ndim != 1 or not np.all(np.isfinite(state)):
            raise ValueError("Cada TaskTurn debe conservar un vector finito.")
        state.setflags(write=False)
        self.latent = state
        self.metadata = dict(self.metadata)


@dataclass
class Task:
    task_id: str
    state: TaskState = TaskState.SUBMITTED
    history: List[TaskTurn] = field(default_factory=list)
    artifacts: Dict = field(default_factory=dict)

    def add_turn(self, latent: np.ndarray, metadata: Optional[Dict] = None) -> TaskTurn:
        t = TaskTurn(
            turn=len(self.history),
            timestamp=datetime.now(timezone.utc).isoformat(),
            latent=latent.copy(),
            hash_latent=hashlib.sha256(latent.tobytes()).hexdigest()[:16],
            metadata=metadata or {})
        self.history.append(t)
        return t


def _json_safe(value):
    """Convierte metadata del prototipo a valores JSON sin deserialización ejecutable."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class SQLiteTaskStore:
    """Almacén SQLite durable: metadatos JSON y vectores float64 en BLOB.

    La deserialización de pickle podría ejecutar código si la base se manipula;
    este MVP persiste estructuras explícitas. Los turnos se insertan de forma
    incremental y no se reescribe el historial en cada avance. SQLite/WAL
    aporta persistencia, no reemplaza una capa de autenticación.
    """
    def __init__(self, db_path: str = "tasks.sqlite3"):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_updated ON tasks(updated_at)")
            conn.execute("""CREATE TABLE IF NOT EXISTS task_turns (
                task_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                latent BLOB NOT NULL,
                latent_dim INTEGER NOT NULL,
                hash_latent TEXT NOT NULL,
                metadata TEXT NOT NULL,
                PRIMARY KEY(task_id, turn),
                FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS checkpoints (
                task_id TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                latent_json TEXT NOT NULL,
                latent_blob BLOB,
                latent_dim INTEGER
            )""")
            checkpoint_columns = {row[1] for row in conn.execute(
                "PRAGMA table_info(checkpoints)").fetchall()}
            if "latent_blob" not in checkpoint_columns:
                conn.execute("ALTER TABLE checkpoints ADD COLUMN latent_blob BLOB")
            if "latent_dim" not in checkpoint_columns:
                conn.execute("ALTER TABLE checkpoints ADD COLUMN latent_dim INTEGER")

    def save(self, task: Task) -> None:
        now = datetime.now(timezone.utc).isoformat()
        created = task.history[0].timestamp if task.history else now
        payload = {
            "task_id": task.task_id,
            "artifacts": _json_safe(task.artifacts),
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=False)
        with self._lock, self._connect() as conn:
            conn.execute("""INSERT INTO tasks(task_id,state,created_at,updated_at,payload)
                VALUES(?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET
                state=excluded.state, updated_at=excluded.updated_at, payload=excluded.payload""",
                         (task.task_id, task.state.value, created, now, encoded))
            row = conn.execute("SELECT COALESCE(MAX(turn), -1) FROM task_turns WHERE task_id=?",
                               (task.task_id,)).fetchone()
            next_turn = int(row[0]) + 1
            # La historia es append-only en Task. Sólo se serializan los turnos
            # nuevos; no se reescriben los vectores anteriores en cada avance.
            for turn in task.history[next_turn:]:
                latent = np.asarray(turn.latent, dtype=np.float64)
                if latent.ndim != 1 or not np.all(np.isfinite(latent)):
                    raise ValueError("La historia contiene un turno latente inválido.")
                metadata_json = json.dumps(_json_safe(turn.metadata),
                                           ensure_ascii=False, separators=(",", ":"),
                                           allow_nan=False)
                conn.execute("""INSERT OR IGNORE INTO task_turns
                    (task_id,turn,timestamp,latent,latent_dim,hash_latent,metadata)
                    VALUES(?,?,?,?,?,?,?)""",
                    (task.task_id, int(turn.turn), turn.timestamp,
                     sqlite3.Binary(latent.tobytes()), int(latent.size),
                     turn.hash_latent, metadata_json))

    def load(self, task_id: str) -> Optional[Task]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT state,payload FROM tasks WHERE task_id=?",
                               (task_id,)).fetchone()
            turns = conn.execute("""SELECT turn,timestamp,latent,latent_dim,
                hash_latent,metadata FROM task_turns WHERE task_id=? ORDER BY turn""",
                (task_id,)).fetchall()
        if row is None:
            return None
        state, payload = row[0], json.loads(row[1])
        task = Task(task_id=payload["task_id"], state=TaskState(state))
        if turns:
            for item in turns:
                latent = np.frombuffer(bytes(item[2]), dtype=np.float64).copy()
                if latent.size != int(item[3]):
                    raise ValueError("El checkpoint de un turno no coincide con su dimensión.")
                task.history.append(TaskTurn(
                    turn=int(item[0]), timestamp=str(item[1]), latent=latent,
                    hash_latent=str(item[4]), metadata=json.loads(item[5])))
        else:
            # Lectura de compatibilidad para bases creadas antes de la tabla
            # incremental; el próximo save migrará sus turnos al formato nuevo.
            for item in payload.get("history", []):
                task.history.append(TaskTurn(
                    turn=int(item["turn"]), timestamp=str(item["timestamp"]),
                    latent=np.asarray(item["latent"], dtype=np.float64),
                    hash_latent=str(item["hash_latent"]),
                    metadata=item.get("metadata", {})))
        task.artifacts = payload.get("artifacts", {})
        return task

    def recover_interrupted(self) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute("""UPDATE tasks SET state=?, updated_at=?
                WHERE state IN (?,?)""",
                (TaskState.FAILED.value, now, TaskState.SUBMITTED.value,
                 TaskState.WORKING.value))
            return int(cursor.rowcount)

    def save_checkpoint(self, task_id: str, latent: np.ndarray) -> None:
        value = np.asarray(latent, dtype=np.float64)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("El checkpoint debe ser un vector finito.")
        with self._lock, self._connect() as conn:
            conn.execute("""INSERT INTO checkpoints
                (task_id,updated_at,latent_json,latent_blob,latent_dim)
                VALUES(?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET
                updated_at=excluded.updated_at, latent_json=excluded.latent_json,
                latent_blob=excluded.latent_blob, latent_dim=excluded.latent_dim""",
                (task_id, datetime.now(timezone.utc).isoformat(), "",
                 sqlite3.Binary(value.tobytes()), int(value.size)))

    def load_checkpoint(self, task_id: str) -> Optional[np.ndarray]:
        with self._lock, self._connect() as conn:
            row = conn.execute("""SELECT latent_json,latent_blob,latent_dim
                               FROM checkpoints WHERE task_id=?""",
                               (task_id,)).fetchone()
        if row is None:
            return None
        if row[1] is not None:
            value = np.frombuffer(bytes(row[1]), dtype=np.float64).copy()
            if row[2] is not None and value.size != int(row[2]):
                raise ValueError("El checkpoint almacenado no coincide con su dimensión.")
            return value
        # Formato anterior a los BLOB: se conserva lectura retrocompatible.
        return np.asarray(json.loads(row[0]), dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# 13. AGENT (con manifest AgentCard para descubrimiento A2A)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class A2AConfig:
    dim: int = 128
    d_prime: int = 6
    connector_mode: str = "hybrid"
    n_tilde: float = 10.0            # régimen del agente en el z-Laplaciano
    ttl: int = 5                     # periodicidad operativa; no derivada de T
    ttl_tol: float = 1e-10
    use_fractal_avg: bool = True
    fractal_noise_std: float = 0.0  # cero conserva identidad y evita trabajo redundante
    seed: int = 0


@dataclass
class _AgentTaskRuntime:
    """Estado Möbius mutable, aislado por tarea para permitir concurrencia."""
    accumulator: np.ndarray
    realigner: TTLRealigner


async def _call_adapter_async(adapter: Callable, *args, **kwargs):
    """Invoca adaptadores sync o async sin bloquear el event loop."""
    is_async = (inspect.iscoroutinefunction(adapter) or
                inspect.iscoroutinefunction(getattr(adapter, "__call__", None)))
    result = adapter(*args, **kwargs) if is_async else await asyncio.to_thread(adapter, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


class Agent:
    """
    Agente con: embedder + conector Möbius + TTL realigner + fractal averager.
    Declara su régimen (ñ, T) vía manifest → el bridge puede elegir Nivel 1 o 2.
    """
    def __init__(self, name: str, embedder: Optional[Callable] = None,
                 config: Optional[A2AConfig] = None, dim: Optional[int] = None,
                 max_active_tasks: int = 4096):
        config = config or A2AConfig()
        if (not math.isfinite(config.n_tilde) or config.n_tilde <= 0 or
                config.ttl < 1 or not math.isfinite(config.fractal_noise_std) or
                config.fractal_noise_std < 0):
            raise ValueError("n_tilde y ttl deben ser positivos; fractal_noise_std no negativo y finito.")
        if max_active_tasks < 1:
            raise ValueError("max_active_tasks debe ser positivo.")
        self.name = name
        self.embedder = embedder
        self.config = config
        self.dim = dim or config.dim
        if config.d_prime > self.dim:
            raise ValueError("d_prime no puede exceder la dimensión del agente.")
        self.connector = MobiusConnector(self.dim, config.d_prime,
                                           config.connector_mode, config.seed)
        self.fractal = FractalAverager(self.connector)
        self.realigner = TTLRealigner(config.ttl, config.ttl_tol)
        self._R_accum = np.eye(config.d_prime)
        self._fitted = False
        self._state_lock = threading.RLock()
        self._task_states: Dict[str, _AgentTaskRuntime] = {}
        self.max_active_tasks = int(max_active_tasks)
        self.T_declared = 1.0 / 7.0 + 4.0 / (7.0 * config.n_tilde)
        if self.embedder is None:
            self.connector.fit_identity()
            self._fitted = True

    def _fit_vectors(self, vectors: np.ndarray) -> 'Agent':
        X = np.asarray(vectors, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dim or len(X) == 0:
            raise ValueError("El embedder no produce la dimensión declarada del agente.")
        if not np.all(np.isfinite(X)):
            raise ValueError("El embedder produjo NaN o infinito.")
        self.connector.fit(X)
        self._fitted = True
        with self._state_lock:
            self._R_accum = np.eye(self.config.d_prime)
            self.realigner.reset()
            self._task_states.clear()
        return self

    def fit(self, corpus: Optional[List[str]] = None) -> 'Agent':
        if self.embedder is None:
            self.connector.fit_identity()
            self._fitted = True
            return self
        if not corpus:
            raise ValueError("El agente necesita un corpus no vacío para ajustar su conector.")
        if (inspect.iscoroutinefunction(self.embedder) or
                inspect.iscoroutinefunction(getattr(self.embedder, "__call__", None))):
            raise RuntimeError("El embedder es asíncrono; usa await Agent.fit_async(corpus).")
        return self._fit_vectors(np.asarray([self.embedder(t) for t in corpus], dtype=np.float64))

    async def fit_async(self, corpus: Optional[List[str]] = None) -> 'Agent':
        """Ajusta el agente con un embedder sync o async, o activa modo latente."""
        if self.embedder is None:
            return self.fit()
        if not corpus:
            raise ValueError("El agente necesita un corpus no vacío para ajustar su conector.")
        semaphore = asyncio.Semaphore(8)

        async def embed_one(text):
            async with semaphore:
                return await _call_adapter_async(self.embedder, text)

        rows = await asyncio.gather(*(embed_one(text) for text in corpus))
        return await asyncio.to_thread(self._fit_vectors, np.asarray(rows, dtype=np.float64))

    def encode(self, text: str) -> np.ndarray:
        """Devuelve el embedding crudo que entra al puente A→B."""
        return self.encode_raw(text)

    def process_latent(self, latent: np.ndarray, task_id: Optional[str] = None) -> np.ndarray:
        """Aplica el rotor del agente a un estado que ya está en su dimensión."""
        if not self._fitted:
            raise RuntimeError(f"El agente {self.name} debe ajustarse antes de procesar estados.")
        v0 = np.asarray(latent, dtype=np.float64)
        if v0.shape != (self.dim,) or not np.all(np.isfinite(v0)):
            raise ValueError(f"El estado entrante debe ser un vector finito de dimensión {self.dim}.")
        R = self.connector.rotor_content(v0)
        with self._state_lock:
            if task_id is None:
                self._R_accum = self.realigner.accumulate(self._R_accum, R)
                accumulated = self._R_accum.copy()
                interaction_count = self.realigner.count
            else:
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError("task_id debe ser una cadena no vacía.")
                runtime = self._task_states.get(task_id)
                if runtime is None:
                    if len(self._task_states) >= self.max_active_tasks:
                        raise RuntimeError("Límite de tareas activas alcanzado; finaliza tareas antes de abrir más.")
                    runtime = _AgentTaskRuntime(
                        np.eye(self.config.d_prime),
                        TTLRealigner(self.config.ttl, self.config.ttl_tol))
                    self._task_states[task_id] = runtime
                runtime.accumulator = runtime.realigner.accumulate(runtime.accumulator, R)
                accumulated = runtime.accumulator.copy()
                interaction_count = runtime.realigner.count
        v1 = self.connector.apply_lift(accumulated, v0)
        if self.config.use_fractal_avg and self.config.fractal_noise_std > 0.0:
            v_final, _ = self.fractal.apply(
                v1, noise_std=self.config.fractal_noise_std,
                seed=self.config.seed + interaction_count)
            return v_final
        return v1

    def begin_task(self, task_id: Optional[str] = None) -> None:
        """Reinicia el acumulador de rotores para una tarea independiente."""
        with self._state_lock:
            if task_id is None:
                self._R_accum = np.eye(self.config.d_prime)
                self.realigner.reset()
            else:
                if task_id not in self._task_states and len(self._task_states) >= self.max_active_tasks:
                    raise RuntimeError("Límite de tareas activas alcanzado; finaliza tareas antes de abrir más.")
                self._task_states[task_id] = _AgentTaskRuntime(
                    np.eye(self.config.d_prime),
                    TTLRealigner(self.config.ttl, self.config.ttl_tol))

    def end_task(self, task_id: str) -> None:
        """Libera el estado geométrico de una tarea terminada o cancelada."""
        with self._state_lock:
            self._task_states.pop(task_id, None)

    def encode_raw(self, text: str) -> np.ndarray:
        """Embedding crudo para calibrar o transportar a través del bridge."""
        if self.embedder is None:
            raise RuntimeError(f"El agente {self.name} no tiene embedder; envía un latent vector.")
        if (inspect.iscoroutinefunction(self.embedder) or
                inspect.iscoroutinefunction(getattr(self.embedder, "__call__", None))):
            raise RuntimeError("El embedder es asíncrono; usa await Agent.encode_raw_async(text).")
        value = np.asarray(self.embedder(text), dtype=np.float64)
        if value.shape != (self.dim,) or not np.all(np.isfinite(value)):
            raise ValueError(f"El embedder de {self.name} devolvió un vector inválido.")
        return value.copy()

    async def encode_raw_async(self, text: str) -> np.ndarray:
        """Codifica texto sin bloquear el event loop; admite embedder async."""
        if self.embedder is None:
            raise RuntimeError(f"El agente {self.name} no tiene embedder; envía un latent vector.")
        value = np.asarray(await _call_adapter_async(self.embedder, text), dtype=np.float64)
        if value.shape != (self.dim,) or not np.all(np.isfinite(value)):
            raise ValueError(f"El embedder de {self.name} devolvió un vector inválido.")
        return value.copy()

    def process_text(self, text: str, task_id: Optional[str] = None) -> np.ndarray:
        return self.process_latent(self.encode_raw(text), task_id=task_id)

    def manifest(self) -> Dict:
        return {
            "name": self.name,
            "version": "0.2.0",
            # JSON-RPC y las partes tensoriales de este archivo son una
            # extensión de prototipo; no se anuncian como conformidad A2A.
            "protocols": ["json-rpc/2.0", "mobius-experimental/0.2"],
            "experimental": True,
            "capabilities": {
                "latent_dims": self.dim,
                "d_prime": self.config.d_prime,
                "n_tilde": self.config.n_tilde,
                "T_declared": self.T_declared,
                "ttl": self.config.ttl,
                "text_embedding": self.embedder is not None,
                "latent_input": True,
                "accepts_parts": ["text", "tensor"] if self.embedder is not None else ["tensor"],
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# 14. SEGURIDAD DE MENSAJES (HMAC, replay guard y política fail-closed)
# ═══════════════════════════════════════════════════════════════════════════

class TrustLevel(Enum):
    UNKNOWN = 0
    HEARD = 1       # firma válida, pero replay/ventana inválida
    TRUSTED = 2


@dataclass
class SecurityVerdict:
    level: TrustLevel
    peer_id: Optional[str]
    reason: str


@dataclass
class AccessDecision:
    allowed: bool
    quarantine: bool
    reason: str


class TrustPolicy:
    """Política explícita; por defecto exige TRUSTED para toda acción."""
    def __init__(self, required_for_encode: TrustLevel = TrustLevel.TRUSTED,
                 required_for_process: TrustLevel = TrustLevel.TRUSTED,
                 quarantine_on_reject: bool = True):
        self.required = {"encode": required_for_encode,
                         "process": required_for_process}
        self.quarantine_on_reject = bool(quarantine_on_reject)

    def check(self, level: TrustLevel, action: str) -> AccessDecision:
        required = self.required.get(action)
        if required is None:
            return AccessDecision(False, self.quarantine_on_reject,
                                  f"acción no autorizada: {action}")
        allowed = level.value >= required.value
        return AccessDecision(allowed, self.quarantine_on_reject and not allowed,
                              "autorizado" if allowed else
                              f"confianza insuficiente: {level.name} < {required.name}")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"solicitud no serializable de forma canónica: {exc}") from exc


def _hkdf_sha256(secret: bytes, info: bytes, length: int = 32) -> bytes:
    """HKDF-SHA256 mínimo para separar claves por protocolo y contexto."""
    salt = b"a2a-mobius/hkdf-sha256/v1"
    prk = hmac.new(salt, secret, hashlib.sha256).digest()
    output = bytearray()
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(prk, previous + info + bytes([counter]), hashlib.sha256).digest()
        output.extend(previous)
        counter += 1
    return bytes(output[:length])


@dataclass(frozen=True)
class SecureEnvelope:
    version: int
    n_tilde: str
    peer_id: str
    nonce: str
    timestamp: str
    tag: str

    @classmethod
    def from_dict(cls, value: Any) -> "SecureEnvelope":
        required = {"version", "n_tilde", "peer_id", "nonce", "timestamp", "tag"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("sobre de seguridad inválido o con campos inesperados.")
        if (type(value["version"]) is not int or value["version"] != 1 or
                not all(isinstance(value[k], str) for k in required - {"version"})):
            raise ValueError("tipos inválidos en el sobre de seguridad.")
        if not value["peer_id"] or len(value["peer_id"]) > 256:
            raise ValueError("peer_id inválido.")
        if len(value["nonce"]) != 32 or any(c not in "0123456789abcdef" for c in value["nonce"]):
            raise ValueError("nonce debe ser 128 bits en hexadecimal minúsculo.")
        if len(value["tag"]) != 64 or any(c not in "0123456789abcdef" for c in value["tag"]):
            raise ValueError("tag HMAC inválido.")
        try:
            t_value = float(value["timestamp"])
            n_value = float(value["n_tilde"])
        except ValueError as exc:
            raise ValueError("timestamp o n_tilde inválido.") from exc
        if not math.isfinite(t_value) or not math.isfinite(n_value) or n_value <= 0:
            raise ValueError("timestamp y n_tilde deben ser finitos; n_tilde debe ser positivo.")
        return cls(value["version"], value["n_tilde"], value["peer_id"],
                   value["nonce"], value["timestamp"], value["tag"])

    def as_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "n_tilde": self.n_tilde,
                "peer_id": self.peer_id, "nonce": self.nonce,
                "timestamp": self.timestamp, "tag": self.tag}


class ScaleToken:
    """Clave HMAC por namespace; firma la solicitud JSON-RPC completa.

    La etiqueta autentica método, id RPC, params y contenido canónico. No
    cifra la comunicación: para confidencialidad hay que habilitar TLS.
    """
    def __init__(self, n_tilde: float, shared_secret: bytes,
                 context: str = "request"):
        if not isinstance(shared_secret, bytes) or len(shared_secret) < 32:
            raise ValueError("shared_secret debe contener al menos 32 bytes.")
        if not isinstance(n_tilde, (int, float)) or not math.isfinite(float(n_tilde)) or n_tilde <= 0:
            raise ValueError("n_tilde debe ser positivo y finito.")
        if not isinstance(context, str) or not context:
            raise ValueError("context debe ser una cadena no vacía.")
        self.n_tilde = float(n_tilde)
        self.namespace = format(self.n_tilde, ".17g")
        self._key = _hkdf_sha256(shared_secret,
                                 f"a2a-mobius/v1/{context}/n={self.namespace}".encode("utf-8"))
        self._response_key = _hkdf_sha256(
            shared_secret, f"a2a-mobius/v1/response/n={self.namespace}".encode("utf-8"))

    @staticmethod
    def _unsigned_request(request: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(request, dict) or not isinstance(request.get("params"), dict):
            raise ValueError("solicitud JSON-RPC inválida para firmar.")
        clean = dict(request)
        params = dict(clean["params"])
        params.pop("security", None)
        clean["params"] = params
        return clean

    def _message(self, request: Dict[str, Any], metadata: Dict[str, Any]) -> bytes:
        unsigned = self._unsigned_request(request)
        return b"A2A-MOBIUS-REQUEST-v1\0" + _canonical_json_bytes(
            {"envelope": metadata, "request": unsigned})

    def sign_request(self, request: Dict[str, Any], peer_id: str,
                     nonce: Optional[str] = None,
                     timestamp: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(peer_id, str) or not peer_id or len(peer_id) > 256:
            raise ValueError("peer_id inválido.")
        nonce = nonce or secrets.token_hex(16)
        timestamp = timestamp or f"{time.time():.6f}"
        metadata = {"version": 1, "n_tilde": self.namespace,
                    "peer_id": peer_id, "nonce": nonce, "timestamp": timestamp}
        tag = hmac.new(self._key, self._message(request, metadata), hashlib.sha256).hexdigest()
        signed = dict(request)
        signed_params = dict(signed["params"])
        signed_params["security"] = {**metadata, "tag": tag}
        signed["params"] = signed_params
        return signed

    def verify_request(self, request: Dict[str, Any], envelope: SecureEnvelope) -> bool:
        if envelope.n_tilde != self.namespace:
            return False
        metadata = {"version": envelope.version, "n_tilde": envelope.n_tilde,
                    "peer_id": envelope.peer_id, "nonce": envelope.nonce,
                    "timestamp": envelope.timestamp}
        expected = hmac.new(self._key, self._message(request, metadata), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, envelope.tag)

    def sign_response(self, response: Dict[str, Any], request_nonce: str) -> Dict[str, Any]:
        """Firma respuesta y la vincula al nonce de la solicitud aceptada."""
        unsigned = dict(response)
        unsigned.pop("security_response", None)
        metadata = {"version": 1, "request_nonce": request_nonce}
        message = b"A2A-MOBIUS-RESPONSE-v1\0" + _canonical_json_bytes(
            {"envelope": metadata, "response": unsigned})
        tag = hmac.new(self._response_key, message, hashlib.sha256).hexdigest()
        signed = dict(unsigned)
        signed["security_response"] = {**metadata, "tag": tag}
        return signed

    def verify_response(self, response: Dict[str, Any], request_nonce: str) -> bool:
        if not isinstance(response, dict):
            return False
        envelope = response.get("security_response")
        if (not isinstance(envelope, dict) or set(envelope) != {"version", "request_nonce", "tag"} or
                envelope.get("version") != 1 or envelope.get("request_nonce") != request_nonce):
            return False
        tag = envelope.get("tag")
        if not isinstance(tag, str) or len(tag) != 64:
            return False
        unsigned = dict(response)
        unsigned.pop("security_response", None)
        message = b"A2A-MOBIUS-RESPONSE-v1\0" + _canonical_json_bytes(
            {"envelope": {"version": 1, "request_nonce": request_nonce},
             "response": unsigned})
        expected = hmac.new(self._response_key, message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, tag)


class ReplayGuard:
    """Replay guard acotado; nunca expulsa nonces todavía vigentes."""
    def __init__(self, window_seconds: float = 30.0,
                 max_entries: int = 100_000, max_peers: int = 10_000):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds debe ser positivo y finito.")
        if max_entries < 1 or max_peers < 1:
            raise ValueError("los límites del ReplayGuard deben ser positivos.")
        self.window_seconds = float(window_seconds)
        self.max_entries = int(max_entries)
        self.max_peers = int(max_peers)
        self._seen: Dict[Tuple[str, str], float] = {}
        self._expiry_heap: List[Tuple[float, Tuple[str, str]]] = []
        self._peer_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def check_and_record(self, peer_id: str, nonce: str, timestamp: str) -> Tuple[bool, str]:
        try:
            sent_at = float(timestamp)
        except (TypeError, ValueError):
            return False, "timestamp inválido"
        now = time.time()
        if not math.isfinite(sent_at) or abs(now - sent_at) > self.window_seconds:
            return False, "timestamp fuera de ventana"
        key = (peer_id, nonce)
        with self._lock:
            while self._expiry_heap and self._expiry_heap[0][0] <= now:
                expiry, item = heapq.heappop(self._expiry_heap)
                if self._seen.get(item) == expiry:
                    del self._seen[item]
                    count = self._peer_counts[item[0]] - 1
                    if count:
                        self._peer_counts[item[0]] = count
                    else:
                        del self._peer_counts[item[0]]
            if key in self._seen:
                return False, "replay detectado"
            if (len(self._seen) >= self.max_entries or
                    (peer_id not in self._peer_counts and len(self._peer_counts) >= self.max_peers)):
                return False, "capacidad de replay guard alcanzada"
            expiry = max(now, sent_at) + self.window_seconds
            self._seen[key] = expiry
            self._peer_counts[peer_id] = self._peer_counts.get(peer_id, 0) + 1
            heapq.heappush(self._expiry_heap, (expiry, key))
            return True, "nonce nuevo"


def verify_incoming(request: Dict[str, Any], expected_token: ScaleToken,
                    replay_guard: ReplayGuard,
                    allowed_peer_ids: Optional[set] = None) -> SecurityVerdict:
    """Valida sobre, identidad permitida, HMAC y nonce antes del handler."""
    params = request.get("params") if isinstance(request, dict) else None
    raw = params.get("security") if isinstance(params, dict) else None
    try:
        envelope = SecureEnvelope.from_dict(raw)
    except ValueError as exc:
        return SecurityVerdict(TrustLevel.UNKNOWN, None, str(exc))
    if allowed_peer_ids is not None and envelope.peer_id not in allowed_peer_ids:
        return SecurityVerdict(TrustLevel.UNKNOWN, envelope.peer_id, "peer_id no autorizado")
    try:
        valid = expected_token.verify_request(request, envelope)
    except (ValueError, TypeError) as exc:
        return SecurityVerdict(TrustLevel.UNKNOWN, envelope.peer_id, f"solicitud inválida: {exc}")
    if not valid:
        return SecurityVerdict(TrustLevel.UNKNOWN, envelope.peer_id, "firma inválida o namespace distinto")
    fresh, reason = replay_guard.check_and_record(envelope.peer_id, envelope.nonce,
                                                  envelope.timestamp)
    if not fresh:
        return SecurityVerdict(TrustLevel.HEARD, envelope.peer_id, reason)
    return SecurityVerdict(TrustLevel.TRUSTED, envelope.peer_id, "solicitud autenticada")


def make_ssl_context(certfile: str, keyfile: str, *,
                     client_ca_file: Optional[str] = None,
                     require_client_cert: bool = False) -> ssl.SSLContext:
    """Crea contexto TLS de servidor; CA de clientes va en archivo separado."""
    if require_client_cert and not client_ca_file:
        raise ValueError("mTLS requiere client_ca_file con la CA de confianza.")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    if require_client_cert:
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(cafile=client_ca_file)
    return context


def compute_peer_id(handler) -> str:
    """Identidad TLS por fingerprint del certificado, si el socket lo expone."""
    try:
        cert_der = handler.connection.getpeercert(binary_form=True)
        if cert_der:
            return "mtls:" + hashlib.sha256(cert_der).hexdigest()
    except (AttributeError, ssl.SSLError):
        pass
    return "anonymous"


class APIKeyAuthenticator:
    """Autenticación Bearer independiente de HMAC y de TLS.

    Sólo conserva SHA-256 de las claves. Las claves pueden venir de
    `A2A_API_KEYS` (separadas por coma) o inyectarse explícitamente.
    """
    def __init__(self, api_keys):
        if isinstance(api_keys, str):
            api_keys = api_keys.split(",")
        keys = [str(key).strip() for key in (api_keys or []) if str(key).strip()]
        if not keys:
            raise ValueError("APIKeyAuthenticator requiere al menos una API key.")
        if any(len(key) < 32 for key in keys):
            raise ValueError("Cada API key debe tener al menos 32 caracteres aleatorios.")
        self._digests = tuple(hashlib.sha256(key.encode("utf-8")).digest() for key in keys)

    @classmethod
    def from_env(cls, env_var: str = "A2A_API_KEYS") -> "APIKeyAuthenticator":
        raw = os.environ.get(env_var, "")
        if not raw.strip():
            raise ValueError(f"Autenticación requerida, pero {env_var} no está configurada.")
        return cls(raw)

    def authenticate(self, authorization: Optional[str]) -> bool:
        if not isinstance(authorization, str):
            return False
        scheme, separator, credential = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not credential:
            return False
        candidate = hashlib.sha256(credential.encode("utf-8")).digest()
        matched = False
        for known in self._digests:
            matched |= hmac.compare_digest(candidate, known)
        return matched


# ═══════════════════════════════════════════════════════════════════════════
# 15. TRANSPORTE HTTP (JSON-RPC A2A extendido con parts tensoriales)
# ═══════════════════════════════════════════════════════════════════════════

def tensor_to_part(v: np.ndarray) -> Dict:
    """Serializa un vector float64 para la extensión tensorial local."""
    value = np.ascontiguousarray(np.asarray(v, dtype=np.float64))
    if value.ndim != 1 or value.size > 12288 or not np.all(np.isfinite(value)):
        raise ValueError("Sólo se transportan vectores float64 finitos de hasta 12,288 elementos.")
    return {"type": "tensor", "dtype": "float64", "shape": list(value.shape),
            "b64": base64.b64encode(value.tobytes()).decode('ascii')}


def part_to_tensor(part: Dict) -> np.ndarray:
    if part.get("dtype") != "float64":
        raise ValueError("El transporte sólo acepta dtype=float64.")
    shape = part.get("shape")
    if (not isinstance(shape, list) or len(shape) != 1 or
            not isinstance(shape[0], int) or not 1 <= shape[0] <= 12288):
        raise ValueError("shape debe describir un vector de longitud 1..12,288.")
    raw = base64.b64decode(part.get("b64", ""), validate=True)
    expected_bytes = shape[0] * np.dtype(np.float64).itemsize
    if len(raw) != expected_bytes:
        raise ValueError("El tamaño del tensor no coincide con shape.")
    result = np.frombuffer(raw, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("El tensor contiene NaN o infinito.")
    return result


class JSONRPCError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = int(code)


class AgentHandler(BaseHTTPRequestHandler):
    """JSON-RPC 2.0 mínimo de este prototipo; no declara conformidad A2A."""

    def log_message(self, *args): pass   # silencio

    def _send_json(self, code: int, obj: Dict) -> None:
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == '/.well-known/agent.json':
            self._send_json(200, self.server.agent.manifest())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != '/rpc':
            return self._send_json(404, {"error": "not found"})
        req = {}
        response_nonce = None
        try:
            authenticator = getattr(self.server, "api_key_authenticator", None)
            if authenticator is not None and not authenticator.authenticate(
                    self.headers.get("Authorization")):
                self.send_response(401)
                self.send_header("WWW-Authenticate", "Bearer")
                self.send_header("Content-Type", "application/json")
                body = b'{"error":"unauthorized"}'
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            n = int(self.headers.get('Content-Length', 0))
            if n <= 0 or n > 8_000_000:
                raise ValueError("Content-Length inválido o mayor que 8 MB.")
            req = json.loads(self.rfile.read(n))
            if not isinstance(req, dict) or req.get("jsonrpc") != "2.0" or "id" not in req:
                raise ValueError("Solicitud JSON-RPC 2.0 inválida.")
            token = getattr(self.server, "security_token", None)
            if token is not None:
                verdict = verify_incoming(
                    req, token, self.server.replay_guard,
                    allowed_peer_ids=self.server.allowed_peer_ids)
                if verdict.level in {TrustLevel.TRUSTED, TrustLevel.HEARD}:
                    response_nonce = req["params"]["security"].get("nonce")
                action = req.get("params", {}).get("action") if isinstance(req.get("params"), dict) else ""
                decision = self.server.trust_policy.check(verdict.level, action)
                if not decision.allowed:
                    # El rechazo no toca estado del agente, tareas ni almacenes.
                    raise JSONRPCError(-32001, f"solicitud rechazada: {verdict.reason}; {decision.reason}")
            result = self._dispatch(req)
            response = {"jsonrpc": "2.0", "id": req.get("id"), "result": result}
            if token is not None:
                response = token.sign_response(response, response_nonce)
            self._send_json(200, response)
        except JSONRPCError as e:
            response = {"jsonrpc": "2.0", "id": req.get("id"),
                        "error": {"code": e.code, "message": str(e)}}
            if response_nonce is not None:
                response = self.server.security_token.sign_response(response, response_nonce)
            self._send_json(200, response)
        except Exception as e:
            response = {"jsonrpc": "2.0", "id": req.get("id"),
                        "error": {"code": -32000, "message": str(e)}}
            if response_nonce is not None and getattr(self.server, "security_token", None) is not None:
                response = self.server.security_token.sign_response(response, response_nonce)
            # JSON-RPC transmite errores de aplicación en el cuerpo. El cliente
            # debe poder leerlos (urllib lanza HTTPError en otros estados).
            self._send_json(200, response)

    def _dispatch(self, req: Dict) -> Dict:
        if req.get("method") == "tasks/send":
            return self._handle_send(req.get("params", {}))
        raise ValueError(f"method no soportado: {req.get('method')}")

    def _handle_send(self, params: Dict) -> Dict:
        if not isinstance(params, dict):
            raise ValueError("params debe ser un objeto.")
        task_id = params.get("task_id")
        action = params.get("action")
        message = params.get("message")
        parts = message.get("parts", []) if isinstance(message, dict) else None
        if not isinstance(task_id, str) or not task_id or not isinstance(parts, list):
            raise ValueError("task_id y message.parts son obligatorios.")
        if not all(isinstance(part, dict) for part in parts):
            raise ValueError("cada parte del mensaje debe ser un objeto.")
        final = params.get("final", True)
        if type(final) is not bool:
            raise ValueError("final debe ser booleano.")
        text_parts = [p for p in parts if p.get("type") == "text"]
        tensor_parts = [p for p in parts if p.get("type") == "tensor"]
        agent = self.server.agent
        if action == "encode":
            if len(text_parts) != 1 or tensor_parts:
                raise ValueError("action=encode requiere exactamente una parte de texto.")
            text = text_parts[0].get("text")
            if not isinstance(text, str):
                raise ValueError("la parte de texto debe contener una cadena.")
            if agent.embedder is not None and (
                    inspect.iscoroutinefunction(agent.embedder) or
                    inspect.iscoroutinefunction(getattr(agent.embedder, "__call__", None))):
                v_out = asyncio.run(agent.encode_raw_async(text))
            else:
                v_out = agent.encode(text)
        elif action == "process":
            if len(tensor_parts) != 1 or text_parts:
                raise ValueError("action=process requiere exactamente una parte tensorial.")
            v_in = part_to_tensor(tensor_parts[0])
            try:
                v_out = agent.process_latent(v_in, task_id=task_id)
            except Exception:
                if final:
                    agent.end_task(task_id)
                raise
        else:
            raise ValueError("action debe ser 'encode' o 'process'.")

        if v_out.shape != (agent.dim,):
            raise ValueError("El agente produjo un estado con dimensión inesperada.")
        task_lock = self.server.task_lock
        with task_lock:
            task = self.server.task_store.get(task_id) or Task(task_id=task_id)
            task.add_turn(v_out, {"role": "server", "action": action})
            task.state = TaskState.COMPLETED
            self.server.task_store[task_id] = task
        if action == "process" and final:
            agent.end_task(task_id)

        # Respuesta: tensor (primario) + texto (compatibilidad A2A estándar)
        return {
            "task_id": task_id,
            "status": task.state.value,
            "turns": len(task.history),
            "message": {
                "role": "agent",
                "parts": [
                    tensor_to_part(v_out),
                    {"type": "text", "text": f"<latent dim={len(v_out)}>"}
                ]
            }
        }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _is_loopback_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def serve(agent: Agent, host: str = "127.0.0.1", port: int = 8080, *,
          security_token: Optional[ScaleToken] = None,
          replay_guard: Optional[ReplayGuard] = None,
          trust_policy: Optional[TrustPolicy] = None,
          allowed_peer_ids: Optional[set] = None,
          api_keys=None, require_api_key: bool = False,
          tls_context: Optional[ssl.SSLContext] = None,
          poll_interval: float = 0.02):
    """Lanza el servidor local de prototipo en un hilo de fondo."""
    poll_interval = float(poll_interval)
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("poll_interval debe ser un número finito positivo.")
    if api_keys is not None:
        api_key_authenticator = APIKeyAuthenticator(api_keys)
    elif require_api_key:
        api_key_authenticator = APIKeyAuthenticator.from_env()
    else:
        api_key_authenticator = None
    if api_key_authenticator is not None and tls_context is None:
        # Bearer credentials must not cross a non-loopback network in cleartext.
        if not _is_loopback_host(host):
            raise ValueError("API keys sobre una red requieren TLS; configura tls_context.")
    httpd = ThreadedHTTPServer((host, port), AgentHandler)
    if security_token is not None and replay_guard is None:
        replay_guard = ReplayGuard()
    httpd.security_token = security_token
    httpd.replay_guard = replay_guard
    httpd.trust_policy = trust_policy or TrustPolicy()
    httpd.allowed_peer_ids = set(allowed_peer_ids) if allowed_peer_ids is not None else None
    httpd.api_key_authenticator = api_key_authenticator
    httpd.tls_enabled = tls_context is not None
    httpd.agent = agent
    httpd.task_store = {}
    httpd.task_lock = threading.RLock()
    if tls_context is not None:
        httpd.socket = tls_context.wrap_socket(httpd.socket, server_side=True)
    t = threading.Thread(target=httpd.serve_forever,
                         kwargs={"poll_interval": poll_interval}, daemon=True)
    t.start()
    return httpd


class A2AClient:
    """Cliente del transporte JSON-RPC experimental definido en este archivo."""
    def __init__(self, base_url: str, timeout: float = 5.0, *,
                 security_token: Optional[ScaleToken] = None,
                 peer_id: str = "protocol-client",
                 api_key: Optional[str] = None,
                 ssl_context: Optional[ssl.SSLContext] = None):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.security_token = security_token
        self.peer_id = peer_id
        self.api_key = api_key
        self.ssl_context = ssl_context
        self._id = 0
        self._id_lock = threading.Lock()

    def discover(self) -> Dict:
        with urllib.request.urlopen(f"{self.base_url}/.well-known/agent.json",
                                    timeout=self.timeout, context=self.ssl_context) as r:
            return json.loads(r.read())

    def send(self, text: Optional[str] = None, latent: Optional[np.ndarray] = None,
             task_id: Optional[str] = None, action: Optional[str] = None,
             final: bool = True) -> Dict:
        if (text is None) == (latent is None):
            raise ValueError("Enviar exactamente uno: text o latent.")
        action = action or ("encode" if text is not None else "process")
        if action not in {"encode", "process"}:
            raise ValueError("action debe ser 'encode' o 'process'.")
        if (action == "encode" and text is None) or (action == "process" and latent is None):
            raise ValueError("encode requiere texto y process requiere un latent vector.")
        if type(final) is not bool:
            raise ValueError("final debe ser booleano.")
        with self._id_lock:
            self._id += 1
            request_id = self._id
        parts = []
        if text is not None:
            parts.append({"type": "text", "text": text})
        if latent is not None:
            parts.append(tensor_to_part(latent))
        payload = {
            "jsonrpc": "2.0", "id": request_id, "method": "tasks/send",
            "params": {"action": action,
                        "task_id": task_id or f"t-{request_id:04d}",
                        "final": final,
                        "message": {"role": "user", "parts": parts}}
        }
        if self.security_token is not None:
            payload = self.security_token.sign_request(payload, self.peer_id)
        req = urllib.request.Request(
            f"{self.base_url}/rpc",
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type': 'application/json'})
        if self.api_key is not None:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout,
                                    context=self.ssl_context) as r:
            result = json.loads(r.read())
        if not isinstance(result, dict):
            raise RuntimeError("El servidor devolvió una respuesta JSON-RPC que no es un objeto.")
        if self.security_token is not None:
            nonce = payload["params"]["security"]["nonce"]
            if "security_response" in result:
                if not self.security_token.verify_response(result, nonce):
                    raise RuntimeError("La respuesta del servidor tiene una firma inválida.")
            elif "error" not in result:
                raise RuntimeError("La respuesta del servidor no tiene una firma válida.")
        if "error" in result:
            error = result["error"]
            message = (error.get("message", "error JSON-RPC")
                       if isinstance(error, dict) else "error JSON-RPC inválido")
            if self.security_token is not None and "security_response" not in result:
                # El mensaje ayuda a diagnosticar errores de autenticación, pero
                # una respuesta sin firma no demuestra que el servidor no actuó.
                raise RuntimeError(
                    f"Error JSON-RPC no autenticado (operación no confirmable): {message}")
            raise RuntimeError(message)
        if result.get("id") != request_id:
            raise RuntimeError("La respuesta JSON-RPC no coincide con la solicitud.")
        return result

    async def discover_async(self) -> Dict:
        """Descubrimiento no bloqueante sobre el cliente HTTP síncrono."""
        return await asyncio.to_thread(self.discover)

    async def send_async(self, text: Optional[str] = None,
                         latent: Optional[np.ndarray] = None,
                         task_id: Optional[str] = None,
                         action: Optional[str] = None,
                         final: bool = True) -> Dict:
        """Envío no bloqueante; mantiene TLS y firma configurados en el cliente."""
        return await asyncio.to_thread(self.send, text, latent, task_id, action, final)


# ═══════════════════════════════════════════════════════════════════════════
# 15. PROTOCOL (orquestador end-to-end)
# ═══════════════════════════════════════════════════════════════════════════

class Protocol:
    """
    Orquesta:
      A.encode → bridge (Nivel 0/1/2) → B.process → DecodeInterface
    Soporta transporte local y un transporte HTTP JSON-RPC experimental.
    """
    def __init__(self, agent_A: Agent, agent_B: Agent,
                 transport: str = "local",
                 host_A: str = "127.0.0.1", port_A: int = 8080,
                 host_B: str = "127.0.0.1", port_B: int = 8081,
                 decoder_fn: Optional[Callable] = None,
                 decoder: Optional[object] = None,
                 store: Optional[Any] = None,
                 aligner: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 security_secret: Optional[bytes] = None,
                 security_peer_id: str = "protocol-client",
                 trust_policy: Optional[TrustPolicy] = None,
                 api_key: Optional[str] = None,
                 api_keys=None,
                 require_api_key: bool = False,
                 rag_signing_key: Optional[bytes] = None,
                 tls_context_A: Optional[ssl.SSLContext] = None,
                 tls_context_B: Optional[ssl.SSLContext] = None,
                 client_ssl_context_A: Optional[ssl.SSLContext] = None,
                 client_ssl_context_B: Optional[ssl.SSLContext] = None,
                 http_poll_interval: float = 0.02):
        self.A, self.B = agent_A, agent_B
        self.transport = transport
        http_poll_interval = float(http_poll_interval)
        if not math.isfinite(http_poll_interval) or http_poll_interval <= 0:
            raise ValueError("http_poll_interval debe ser un número finito positivo.")
        self.http_poll_interval = http_poll_interval
        if transport not in {"local", "http"}:
            raise ValueError("transport debe ser 'local' o 'http'.")
        if decoder is not None and decoder_fn is not None:
            raise ValueError("Pasa decoder o decoder_fn, no ambos.")
        if decoder is not None:
            if hasattr(decoder, "decode"):
                self.decoder = decoder
            elif callable(decoder):
                self.decoder = DecodeInterface(decoder)
            else:
                raise TypeError("decoder debe ser invocable o exponer decode(vector).")
        else:
            self.decoder = DecodeInterface(decoder_fn)
        if rag_signing_key is not None and (
                not isinstance(rag_signing_key, bytes) or len(rag_signing_key) < 32):
            raise ValueError("rag_signing_key debe contener al menos 32 bytes.")
        self.rag_signing_key = rag_signing_key
        # Si el índice ya existe, se firma al construir el protocolo. Si se
        # llenará con calibrate(), se firma inmediatamente después de indexar.
        if (self.rag_signing_key is not None and
                isinstance(self.decoder, RAGDecodeInterface) and
                self.decoder.corpus_emb is not None):
            self.decoder = SecureRAGIndex(self.decoder, self.rag_signing_key)
        self.store = store
        self.recovered_tasks = 0
        self._store_recovery_pending = False
        recovery = getattr(self.store, "recover_interrupted", None) if self.store is not None else None
        if callable(recovery):
            if (inspect.iscoroutinefunction(recovery) or
                    inspect.iscoroutinefunction(getattr(recovery, "__call__", None))):
                self._store_recovery_pending = True
            else:
                recovered = recovery()
                if inspect.isawaitable(recovered):
                    # No se puede esperar desde __init__. Cerramos la coroutine
                    # sin ejecutarla y la ejecutamos en el primer send_async().
                    close_pending = getattr(recovered, "close", None)
                    if callable(close_pending):
                        close_pending()
                    self._store_recovery_pending = True
                else:
                    self.recovered_tasks = int(recovered or 0)
        self.servers: List = []
        self.client_A: Optional[A2AClient] = None
        self.client_B: Optional[A2AClient] = None
        self.task_count = 0
        self._task_count_lock = threading.Lock()
        self.bridge_meta: Dict = {"level": "identity", "bridge_representation": "same-space identity"}
        self._bridge_meta_public: Dict = {"level": "identity", "bridge_representation": "same-space identity"}
        self._align_fn: Optional[Callable[[np.ndarray], np.ndarray]] = aligner
        if self._align_fn is None and agent_A.dim == agent_B.dim:
            self._align_fn = lambda v: np.asarray(v, dtype=np.float64).copy()
        self.security_peer_id = security_peer_id
        self.trust_policy = trust_policy or TrustPolicy()
        self._token_A = ScaleToken(agent_A.config.n_tilde, security_secret) if security_secret is not None else None
        self._token_B = ScaleToken(agent_B.config.n_tilde, security_secret) if security_secret is not None else None
        self._replay_guard_A = ReplayGuard()
        self._replay_guard_B = ReplayGuard()
        server_api_keys = api_keys
        if require_api_key and server_api_keys is None:
            if api_key is not None:
                server_api_keys = [api_key]
            else:
                raw_keys = os.environ.get("A2A_API_KEYS", "")
                server_api_keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        elif server_api_keys is None and api_key is not None:
            # En este orquestador cliente/servidor, una única api_key explícita
            # protege automáticamente los endpoints HTTP que él mismo crea.
            server_api_keys = [api_key]
        if isinstance(server_api_keys, str):
            server_api_keys = [key.strip() for key in server_api_keys.split(",") if key.strip()]
        if api_key is None and server_api_keys and len(server_api_keys) == 1:
            api_key = server_api_keys[0]
        if (api_key is not None and server_api_keys is not None and
                api_key not in server_api_keys):
            raise ValueError("api_key debe estar incluida en api_keys aceptadas por el servidor.")
        if transport == "http" and (require_api_key or server_api_keys) and api_key is None:
            raise ValueError("Configura api_key para el cliente interno del transporte HTTP protegido.")
        self.api_key = api_key
        self.api_keys = server_api_keys
        self.require_api_key = bool(require_api_key)
        self.tls_context_A = tls_context_A
        self.tls_context_B = tls_context_B
        self.client_ssl_context_A = client_ssl_context_A
        self.client_ssl_context_B = client_ssl_context_B

        if transport == "http":
            if server_api_keys:
                for host, context in ((host_A, tls_context_A), (host_B, tls_context_B)):
                    if context is None and not _is_loopback_host(host):
                        raise ValueError(
                            "API keys sobre una red requieren TLS; configura tls_context_A/B.")
            server_a = serve(agent_A, host_A, port_A,
                             security_token=self._token_A,
                             replay_guard=self._replay_guard_A,
                             trust_policy=self.trust_policy,
                             allowed_peer_ids={security_peer_id} if self._token_A else None,
                             api_keys=server_api_keys, require_api_key=require_api_key,
                             tls_context=tls_context_A,
                             poll_interval=self.http_poll_interval)
            server_b = serve(agent_B, host_B, port_B,
                             security_token=self._token_B,
                             replay_guard=self._replay_guard_B,
                             trust_policy=self.trust_policy,
                             allowed_peer_ids={security_peer_id} if self._token_B else None,
                             api_keys=server_api_keys, require_api_key=require_api_key,
                             tls_context=tls_context_B,
                             poll_interval=self.http_poll_interval)
            self.servers.extend([server_a, server_b])
            scheme_A = "https" if tls_context_A is not None else "http"
            scheme_B = "https" if tls_context_B is not None else "http"
            self.client_A = A2AClient(
                f"{scheme_A}://{host_A}:{server_a.server_port}",
                security_token=self._token_A, peer_id=security_peer_id,
                api_key=api_key, ssl_context=client_ssl_context_A)
            self.client_B = A2AClient(
                f"{scheme_B}://{host_B}:{server_b.server_port}",
                security_token=self._token_B, peer_id=security_peer_id,
                api_key=api_key, ssl_context=client_ssl_context_B)

    @property
    def endpoints(self) -> Dict[str, str]:
        """Direcciones realmente asignadas, útil si se pidió un puerto efímero."""
        return {server.agent.name:
                f"{'https' if getattr(server, 'tls_enabled', False) else 'http'}://"
                f"{server.server_address[0]}:{server.server_port}"
                for server in self.servers}

    def calibrate(self, corpus: List[str],
                  corpus_ids: Optional[List[str]] = None,
                  nested_coordinates: bool = False,
                  max_bridge_rank: Optional[int] = 128,
                  max_kernel_anchors: Optional[int] = 2000,
                  svd_oversamples: int = 10,
                  svd_power_iterations: int = 4) -> 'Protocol':
        """
        Calibra ambos conectores y ajusta el bridge con embeddings pareados.
        La selección z/log/KRR se evalúa en anclas reservadas cuando el corpus
        tiene tamaño suficiente; el valor T sólo es metadato y una preferencia.
        `max_bridge_rank=None` desactiva el truncamiento SVD y
        `max_kernel_anchors=None` permite entrenar Kernel Ridge con todas las anclas.
        Los parámetros `svd_oversamples` y `svd_power_iterations` ajustan
        precisión y costo del bridge aleatorio.
        """
        if self.A.embedder is None or self.B.embedder is None:
            raise RuntimeError("calibrate(corpus) requiere embedders; para usarlos fuera del protocolo, envía latent=... con dimensiones compatibles.")
        self.A.fit(corpus)
        self.B.fit(corpus)

        # Verificación z-Laplace
        self.z_checks = {
            name: verify_z_laplacian(ag.T_declared, ag.config.n_tilde)
            for name, ag in (("A", self.A), ("B", self.B))
        }

        # Bridge con regla secuencial
        X_A = np.array([self.A.encode_raw(t) for t in corpus])
        X_B = np.array([self.B.encode_raw(t) for t in corpus])
        align_fn, meta = build_bridge(X_A, X_B, self.A.T_declared,
                                      self.B.T_declared, seed=0,
                                      nested_coordinates=nested_coordinates,
                                      max_bridge_rank=max_bridge_rank,
                                      max_kernel_anchors=max_kernel_anchors,
                                      svd_oversamples=svd_oversamples,
                                      svd_power_iterations=svd_power_iterations)
        self._align_fn = align_fn
        self.bridge_meta = meta
        self._bridge_meta_public = {k: v for k, v in meta.items() if k != "bridge"}

        # Si se pasó un RAG vacío, indexa ejemplos procesados por el mismo
        # puente y el mismo agente B que se usarán en consultas posteriores.
        rag = self._rag_interface()
        if rag is not None and rag.corpus_emb is None:
            rag_vectors = []
            for text in corpus:
                self.B.begin_task()
                mapped = self._align_fn(self.A.encode_raw(text))
                rag_vectors.append(self.B.process_latent(mapped))
            self.B.begin_task()
            rag.fit_index(np.asarray(rag_vectors), corpus, corpus_ids)
            self._secure_rag_if_configured(rag)
        return self

    async def calibrate_async(self, corpus: List[str],
                              corpus_ids: Optional[List[str]] = None,
                              nested_coordinates: bool = False,
                              max_bridge_rank: Optional[int] = 128,
                              max_kernel_anchors: Optional[int] = 2000,
                              svd_oversamples: int = 10,
                              svd_power_iterations: int = 4) -> 'Protocol':
        """Calibra sin bloquear el event loop; admite embedders async/sync."""
        if self.A.embedder is None or self.B.embedder is None:
            raise RuntimeError("calibrate_async(corpus) requiere embedders; el modo sin embeddings usa latent=... y no necesita calibración si las dimensiones coinciden.")

        async def embed_corpus(agent: Agent):
            semaphore = asyncio.Semaphore(8)

            async def one(text):
                async with semaphore:
                    return await agent.encode_raw_async(text)

            return np.asarray(await asyncio.gather(*(one(text) for text in corpus)),
                              dtype=np.float64)

        X_A, X_B = await asyncio.gather(embed_corpus(self.A), embed_corpus(self.B))
        await asyncio.gather(
            asyncio.to_thread(self.A._fit_vectors, X_A),
            asyncio.to_thread(self.B._fit_vectors, X_B))
        self.z_checks = {
            name: verify_z_laplacian(agent.T_declared, agent.config.n_tilde)
            for name, agent in (("A", self.A), ("B", self.B))
        }
        align_fn, meta = await asyncio.to_thread(
            build_bridge, X_A, X_B, self.A.T_declared, self.B.T_declared,
            seed=0, nested_coordinates=nested_coordinates,
            max_bridge_rank=max_bridge_rank,
            max_kernel_anchors=max_kernel_anchors,
            svd_oversamples=svd_oversamples,
            svd_power_iterations=svd_power_iterations)
        self._align_fn = align_fn
        self.bridge_meta = meta
        self._bridge_meta_public = {key: value for key, value in meta.items()
                                    if key != "bridge"}

        rag = self._rag_interface()
        if rag is not None and rag.corpus_emb is None:
            rag_vectors = []
            for index, vector in enumerate(X_A):
                task_id = f"calibration-rag-{index}"
                mapped = self._align_fn(vector)
                rag_vectors.append(self.B.process_latent(mapped, task_id=task_id))
                self.B.end_task(task_id)
            await asyncio.to_thread(rag.fit_index,
                                    np.asarray(rag_vectors), corpus, corpus_ids)
            self._secure_rag_if_configured(rag)
        return self

    def _rag_interface(self) -> Optional[RAGDecodeInterface]:
        """Devuelve el RAG incluso cuando Protocol lo envuelve con firma."""
        if isinstance(self.decoder, SecureRAGIndex):
            return self.decoder.rag
        return self.decoder if isinstance(self.decoder, RAGDecodeInterface) else None

    def _secure_rag_if_configured(self, rag: RAGDecodeInterface) -> None:
        """Firma el índice ya ajustado sin sustituir referencias del usuario."""
        if self.rag_signing_key is not None and self.decoder is rag:
            self.decoder = SecureRAGIndex(rag, self.rag_signing_key)

    def _next_task_id(self) -> str:
        with self._task_count_lock:
            self.task_count += 1
            return f"task-{self.task_count:04d}-{uuid.uuid4().hex[:12]}"

    def _ensure_aligner(self) -> Callable[[np.ndarray], np.ndarray]:
        if self._align_fn is None and self.A.dim == self.B.dim:
            self._align_fn = lambda v: np.asarray(v, dtype=np.float64).copy()
            self.bridge_meta = {"level": "identity", "bridge_representation": "same-space identity"}
            self._bridge_meta_public = dict(self.bridge_meta)
        if self._align_fn is None:
            raise RuntimeError("Falta un bridge A→B: calibra con embeddings o inyecta aligner. Las dimensiones distintas no se adivinan.")
        return self._align_fn

    def _validate_input(self, text: Optional[str], latent: Optional[np.ndarray]) -> None:
        if (text is None) == (latent is None):
            raise ValueError("Envía exactamente uno: text o latent.")
        if text is not None and not isinstance(text, str):
            raise TypeError("text debe ser str; para vectores usa latent=...")
        if latent is not None:
            value = np.asarray(latent, dtype=np.float64)
            if value.shape != (self.A.dim,) or not np.all(np.isfinite(value)):
                raise ValueError(f"latent debe ser un vector finito de dimensión {self.A.dim}.")

    def _apply_bridge(self, vector: np.ndarray) -> np.ndarray:
        aligner = self._ensure_aligner()
        if (inspect.iscoroutinefunction(aligner) or
                inspect.iscoroutinefunction(getattr(aligner, "__call__", None))):
            raise RuntimeError("aligner es asíncrono; usa await Protocol.send_async(...).")
        mapped = np.asarray(aligner(np.asarray(vector, dtype=np.float64).copy()), dtype=np.float64)
        if mapped.shape != (self.B.dim,) or not np.all(np.isfinite(mapped)):
            raise ValueError(f"aligner debe devolver un vector finito de dimensión {self.B.dim}.")
        return mapped.copy()

    async def _apply_bridge_async(self, vector: np.ndarray) -> np.ndarray:
        aligner = self._ensure_aligner()
        mapped = np.asarray(await _call_adapter_async(
            aligner, np.asarray(vector, dtype=np.float64).copy()), dtype=np.float64)
        if mapped.shape != (self.B.dim,) or not np.all(np.isfinite(mapped)):
            raise ValueError(f"aligner debe devolver un vector finito de dimensión {self.B.dim}.")
        return mapped.copy()

    def _save_task(self, task: Task, checkpoint: Optional[np.ndarray] = None) -> None:
        if self.store is None:
            return
        save = getattr(self.store, "save", None)
        if callable(save):
            if inspect.iscoroutinefunction(save):
                raise RuntimeError("El almacén es asíncrono; usa await Protocol.send_async(...).")
            result = save(task)
            if inspect.isawaitable(result):
                raise RuntimeError("El almacén devolvió un awaitable; usa Protocol.send_async(...).")
        checkpoint_fn = getattr(self.store, "save_checkpoint", None)
        if checkpoint is not None and callable(checkpoint_fn):
            if inspect.iscoroutinefunction(checkpoint_fn):
                raise RuntimeError("save_checkpoint es asíncrono; usa Protocol.send_async(...).")
            result = checkpoint_fn(task.task_id, checkpoint)
            if inspect.isawaitable(result):
                raise RuntimeError("save_checkpoint devolvió un awaitable; usa Protocol.send_async(...).")

    async def _recover_store_async(self) -> None:
        if not self._store_recovery_pending or self.store is None:
            return
        recovery = getattr(self.store, "recover_interrupted", None)
        if callable(recovery):
            self.recovered_tasks = int(await _call_adapter_async(recovery) or 0)
        self._store_recovery_pending = False

    async def _save_task_async(self, task: Task,
                               checkpoint: Optional[np.ndarray] = None) -> None:
        if self.store is None:
            return
        save = getattr(self.store, "save", None)
        if callable(save):
            await _call_adapter_async(save, task)
        checkpoint_fn = getattr(self.store, "save_checkpoint", None)
        if checkpoint is not None and callable(checkpoint_fn):
            await _call_adapter_async(checkpoint_fn, task.task_id, checkpoint)

    def _decode_sync(self, vector: np.ndarray) -> Dict:
        decoder_fn = getattr(self.decoder, "decoder_fn", None)
        if decoder_fn is not None and (
                inspect.iscoroutinefunction(decoder_fn) or
                inspect.iscoroutinefunction(getattr(decoder_fn, "__call__", None))):
            raise RuntimeError("El decoder es asíncrono; usa await Protocol.send_async(...).")
        decode = getattr(self.decoder, "decode", None)
        if not callable(decode):
            raise TypeError("decoder debe exponer decode(vector).")
        if inspect.iscoroutinefunction(decode):
            raise RuntimeError("El decoder es asíncrono; usa await Protocol.send_async(...).")
        result = decode(np.asarray(vector, dtype=np.float64).copy())
        if inspect.isawaitable(result):
            raise RuntimeError("El decoder devolvió un awaitable; usa Protocol.send_async(...).")
        return result

    async def _decode_async(self, vector: np.ndarray) -> Dict:
        decoder_fn = getattr(self.decoder, "decoder_fn", None)
        if decoder_fn is not None:
            original = np.asarray(vector, dtype=np.float64)
            text = await _call_adapter_async(decoder_fn, original.copy())
            return {"text": text,
                    "hash_original": hashlib.sha256(original.tobytes()).hexdigest()[:16],
                    "norm_original": float(np.linalg.norm(original))}
        decode = getattr(self.decoder, "decode", None)
        if not callable(decode):
            raise TypeError("decoder debe exponer decode(vector).")
        return await _call_adapter_async(decode, np.asarray(vector, dtype=np.float64).copy())

    def send(self, text: Optional[str] = None, turns: int = 1, *,
             latent: Optional[np.ndarray] = None,
             task_id: Optional[str] = None) -> Any:
        """Flujo síncrono; acepta texto o vector externo, sin DB obligatoria."""
        if self._store_recovery_pending:
            raise RuntimeError("La recuperación del almacén es asíncrona; usa await Protocol.send_async(...).")
        if turns < 1:
            raise ValueError("turns debe ser al menos 1.")
        self._validate_input(text, latent)
        task_id = task_id or self._next_task_id()
        if self.transport == "local":
            return self._send_local(text, latent, turns, task_id)
        return self._send_http(text, latent, turns, task_id)

    async def send_async(self, text: Optional[str] = None, turns: int = 1, *,
                         latent: Optional[np.ndarray] = None,
                         task_id: Optional[str] = None) -> Any:
        """Flujo async con soporte de adapters async y persistencia opcional."""
        if turns < 1:
            raise ValueError("turns debe ser al menos 1.")
        self._validate_input(text, latent)
        await self._recover_store_async()
        task_id = task_id or self._next_task_id()
        if self.transport == "local":
            return await self._send_local_async(text, latent, turns, task_id)
        return await self._send_http_async(text, latent, turns, task_id)

    def _send_local(self, text: Optional[str], latent: Optional[np.ndarray],
                    turns: int, task_id: str) -> Task:
        task = Task(task_id=task_id, state=TaskState.WORKING)
        self.B.begin_task(task_id)
        try:
            self._save_task(task)
            v_A = self.A.encode(text) if text is not None else np.asarray(latent, dtype=np.float64).copy()
            task.add_turn(v_A, {"role": "A" if text is not None else "external-latent"})
            self._save_task(task, v_A)
            v_B = self._apply_bridge(v_A)
            task.add_turn(v_B, {"role": "bridge", **self._bridge_meta_public})
            self._save_task(task, v_B)
            for turn in range(turns):
                v_B = self.B.process_latent(v_B, task_id=task_id)
                task.add_turn(v_B, {"role": "B", "turn": turn})
                self._save_task(task, v_B)
            task.artifacts["decoded"] = self._decode_sync(v_B)
            task.state = TaskState.COMPLETED
            self._save_task(task, v_B)
            return task
        except Exception:
            task.state = TaskState.FAILED
            self._save_task(task)
            raise
        finally:
            self.B.end_task(task_id)

    async def _send_local_async(self, text: Optional[str], latent: Optional[np.ndarray],
                                turns: int, task_id: str) -> Task:
        task = Task(task_id=task_id, state=TaskState.WORKING)
        self.B.begin_task(task_id)
        try:
            await self._save_task_async(task)
            v_A = (await self.A.encode_raw_async(text) if text is not None else
                   np.asarray(latent, dtype=np.float64).copy())
            task.add_turn(v_A, {"role": "A" if text is not None else "external-latent"})
            await self._save_task_async(task, v_A)
            v_B = await self._apply_bridge_async(v_A)
            task.add_turn(v_B, {"role": "bridge", **self._bridge_meta_public})
            await self._save_task_async(task, v_B)
            for turn in range(turns):
                v_B = await asyncio.to_thread(self.B.process_latent, v_B, task_id)
                task.add_turn(v_B, {"role": "B", "turn": turn})
                await self._save_task_async(task, v_B)
            task.artifacts["decoded"] = await self._decode_async(v_B)
            task.state = TaskState.COMPLETED
            await self._save_task_async(task, v_B)
            return task
        except Exception:
            task.state = TaskState.FAILED
            await self._save_task_async(task)
            raise
        finally:
            self.B.end_task(task_id)

    def _send_http(self, text: Optional[str], latent: Optional[np.ndarray],
                   turns: int, task_id: str) -> Dict:
        if self.client_A is None or self.client_B is None:
            raise RuntimeError("El transporte HTTP no está inicializado.")
        task = Task(task_id=task_id, state=TaskState.WORKING)
        self._save_task(task)
        if text is not None:
            resp_A = self.client_A.send(text=text, task_id=f"{task_id}-A", action="encode")
            v_A = part_to_tensor(resp_A["result"]["message"]["parts"][0])
            input_role = "A"
        else:
            v_A = np.asarray(latent, dtype=np.float64).copy()
            input_role = "external-latent"
        task.add_turn(v_A, {"role": input_role, "transport": "http"})
        self._save_task(task, v_A)
        v_B = self._apply_bridge(v_A)
        task.add_turn(v_B, {"role": "bridge", **self._bridge_meta_public})
        self._save_task(task, v_B)
        resp = None
        b_task_id = f"{task_id}-B"
        for turn in range(turns):
            resp = self.client_B.send(latent=v_B, task_id=b_task_id, action="process",
                                      final=(turn == turns - 1))
            v_B = part_to_tensor(resp["result"]["message"]["parts"][0])
            task.add_turn(v_B, {"role": "B", "turn": turn, "transport": "http"})
            self._save_task(task, v_B)
        decoded = self._decode_sync(v_B)
        task.artifacts["decoded"] = decoded
        task.state = TaskState.COMPLETED
        self._save_task(task, v_B)
        return {"task_id": task_id, "state": task.state.value,
                "turns_http": turns, "response": resp, "decoded": decoded,
                "bridge_meta": dict(self._bridge_meta_public)}

    async def _send_http_async(self, text: Optional[str], latent: Optional[np.ndarray],
                               turns: int, task_id: str) -> Dict:
        if self.client_A is None or self.client_B is None:
            raise RuntimeError("El transporte HTTP no está inicializado.")
        task = Task(task_id=task_id, state=TaskState.WORKING)
        await self._save_task_async(task)
        if text is not None:
            resp_A = await self.client_A.send_async(text=text, task_id=f"{task_id}-A",
                                                    action="encode")
            v_A = part_to_tensor(resp_A["result"]["message"]["parts"][0])
            input_role = "A"
        else:
            v_A = np.asarray(latent, dtype=np.float64).copy()
            input_role = "external-latent"
        task.add_turn(v_A, {"role": input_role, "transport": "http"})
        await self._save_task_async(task, v_A)
        v_B = await self._apply_bridge_async(v_A)
        task.add_turn(v_B, {"role": "bridge", **self._bridge_meta_public})
        await self._save_task_async(task, v_B)
        resp = None
        b_task_id = f"{task_id}-B"
        for turn in range(turns):
            resp = await self.client_B.send_async(
                latent=v_B, task_id=b_task_id, action="process",
                final=(turn == turns - 1))
            v_B = part_to_tensor(resp["result"]["message"]["parts"][0])
            task.add_turn(v_B, {"role": "B", "turn": turn, "transport": "http"})
            await self._save_task_async(task, v_B)
        decoded = await self._decode_async(v_B)
        task.artifacts["decoded"] = decoded
        task.state = TaskState.COMPLETED
        await self._save_task_async(task, v_B)
        return {"task_id": task_id, "state": task.state.value,
                "turns_http": turns, "response": resp, "decoded": decoded,
                "bridge_meta": dict(self._bridge_meta_public)}

    def close(self) -> None:
        """Detiene y libera los servidores HTTP lanzados por este protocolo."""
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.servers.clear()

    async def close_async(self) -> None:
        """Cierra servidores sin bloquear el event loop."""
        await asyncio.to_thread(self.close)


# ═══════════════════════════════════════════════════════════════════════════
# 15A. CHAT OPCIONAL (LiteLLM + Gradio, imports perezosos)
# ═══════════════════════════════════════════════════════════════════════════

class LiteLLMChatGateway:
    """Adaptador opcional para modelos de chat vía LiteLLM.

    LiteLLM y las claves del proveedor no son requisitos del protocolo. Las
    credenciales se leen de las variables que define cada proveedor (por ej.,
    OPENAI_API_KEY o ANTHROPIC_API_KEY); nunca se guardan en este objeto.
    """
    def __init__(self, model: str, *, api_base: Optional[str] = None,
                 temperature: float = 0.2, timeout: float = 60.0):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model debe ser un identificador LiteLLM, por ejemplo provider/model.")
        if not math.isfinite(temperature) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("temperature y timeout deben ser finitos; timeout debe ser positivo.")
        self.model = model.strip()
        self.api_base = api_base
        self.temperature = float(temperature)
        self.timeout = float(timeout)

    @staticmethod
    def _response_text(response) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("LiteLLM devolvió una respuesta sin choices[0].message.content.") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content
                           if isinstance(part, dict) and part.get("type") == "text")
        raise RuntimeError("LiteLLM devolvió contenido no textual; configura un adaptador multimodal.")

    async def complete_async(self, messages: List[Dict[str, str]]) -> str:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError("Instala LiteLLM para habilitar este gateway (pip install litellm).") from exc
        options = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "timeout": self.timeout}
        if self.api_base is not None:
            options["api_base"] = self.api_base
        response = await litellm.acompletion(**options)
        return self._response_text(response)

    def complete(self, messages: List[Dict[str, str]]) -> str:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError("Instala LiteLLM para habilitar este gateway (pip install litellm).") from exc
        options = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "timeout": self.timeout}
        if self.api_base is not None:
            options["api_base"] = self.api_base
        response = litellm.completion(**options)
        return self._response_text(response)


class GradioChatAdapter:
    """UI Gradio opcional conectada a un gateway o al Protocol.

    Se debe elegir un backend. Con gateway, el chat es LLM convencional. Con
    Protocol, el adapter devuelve el resultado del decoder; si el flujo opera
    con vectores externos, `input_encoder` es el adaptador explícito texto→vector.
    """
    def __init__(self, *, gateway: Optional[LiteLLMChatGateway] = None,
                 protocol: Optional[Protocol] = None,
                 input_encoder: Optional[Callable[[str], np.ndarray]] = None,
                 system_prompt: Optional[str] = None,
                 history_limit: int = 16):
        if (gateway is None) == (protocol is None):
            raise ValueError("Elige exactamente uno: gateway o protocol.")
        if history_limit < 0:
            raise ValueError("history_limit debe ser no negativo.")
        self.gateway = gateway
        self.protocol = protocol
        self.input_encoder = input_encoder
        self.system_prompt = system_prompt
        self.history_limit = int(history_limit)

    @staticmethod
    def _content_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content
                           if isinstance(part, dict) and part.get("type") == "text")
        return ""

    def _messages(self, message: str, history) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        recent = list(history or [])
        recent = recent[-self.history_limit:] if self.history_limit else []
        for item in recent:
            if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
                content = self._content_text(item.get("content"))
                if content:
                    messages.append({"role": item["role"], "content": content})
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                # Compatibilidad con el formato de historial por pares de
                # versiones anteriores de Gradio.
                user_text, assistant_text = item
                if isinstance(user_text, str) and user_text:
                    messages.append({"role": "user", "content": user_text})
                if isinstance(assistant_text, str) and assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": message})
        return messages

    async def respond(self, message: str, history=None) -> str:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("El mensaje del chat no puede estar vacío.")
        if self.gateway is not None:
            return await self.gateway.complete_async(self._messages(message, history))
        if self.input_encoder is None:
            result = await self.protocol.send_async(text=message)
        else:
            latent = await _call_adapter_async(self.input_encoder, message)
            result = await self.protocol.send_async(latent=np.asarray(latent, dtype=np.float64))
        if isinstance(result, Task):
            decoded = result.artifacts.get("decoded", {})
        else:
            decoded = result.get("decoded", {}) if isinstance(result, dict) else {}
        text = decoded.get("text") if isinstance(decoded, dict) else None
        if not isinstance(text, str):
            raise RuntimeError("El protocolo no devolvió texto; configura un decoder para mostrarlo en el chat.")
        return text

    def build(self, *, title: str = "Chat con AlphA",
              description: str = "Interfaz opcional sobre el monolito modular.",
              **chat_interface_options):
        """Construye ChatInterface; Gradio sólo se importa cuando se solicita."""
        try:
            import gradio as gr
        except ImportError as exc:
            raise RuntimeError("Instala Gradio para habilitar la interfaz (pip install gradio).") from exc
        return gr.ChatInterface(fn=self.respond, title=title,
                                description=description, **chat_interface_options)

    def launch(self, **launch_options):
        """Atajo para construir y abrir la UI cuando Gradio está instalado."""
        return self.build().launch(**launch_options)


# ═══════════════════════════════════════════════════════════════════════════
# 16. DIAGNÓSTICO (ratio vs. baseline aleatorio)
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_alignment(X: np.ndarray, Y: np.ndarray) -> Dict:
    """
    Compara errores en filas reservadas para pares reales y etiquetas permutadas.
    Los umbrales del veredicto son heurísticos y no una prueba de ausencia o
    existencia de relación semántica.
    """
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y) or len(X) < 2:
        raise ValueError("diagnose_alignment requiere al menos dos pares de filas.")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(X))
    n_val = max(1, len(X) // 4) if len(X) >= 5 else 0
    val_idx, train_idx = order[:n_val], order[n_val:]
    if n_val == 0:
        train_idx, val_idx = order, order
    b = StateBridge().fit(X[train_idx], Y[train_idx])
    err_real = float(np.linalg.norm(b.align(X[val_idx]) - Y[val_idx]) /
                      max(np.linalg.norm(Y[val_idx]), 1e-15))

    Y_train = Y[train_idx]
    Y_perm_train = Y_train[rng.permutation(len(Y_train))]
    b2 = StateBridge().fit(X[train_idx], Y_perm_train)
    err_perm = float(np.linalg.norm(b2.align(X[val_idx]) - Y[val_idx]) /
                     max(np.linalg.norm(Y[val_idx]), 1e-15))

    ratio = err_real / (err_perm + 1e-15)
    verdict = ("✅ señal clara" if ratio < 0.5 else
               "⚠️  señal débil" if ratio < 0.8 else
               "❌ sin señal")
    return {"err_real": err_real, "err_perm": err_perm,
            "ratio": ratio, "verdict": verdict,
            "validation_count": int(n_val), "selection_in_sample": n_val == 0}


# ═══════════════════════════════════════════════════════════════════════════
# 17. TESTS (ground truth + base compartida + warp de frecuencia + end-to-end)
# ═══════════════════════════════════════════════════════════════════════════

def test_randomized_svd_equivalence():
    """Comprueba SVD aleatoria, PCA de MobiusConnector y StateBridge truncado."""
    rng = np.random.default_rng(7)
    N, D, k = 80, 400, 6
    latent = rng.standard_normal((N, k))
    frame, _ = qr_twice(rng.standard_normal((D, k)))
    matrix = latent @ frame.T + 1e-9 * rng.standard_normal((N, D))
    _, singular_full, Vt_full = np.linalg.svd(matrix, full_matrices=False)
    _, singular_truncated, Vt_truncated = _randomized_svd(
        matrix, k, seed=0)

    projection_full = Vt_full[:k].T @ Vt_full[:k]
    projection_truncated = Vt_truncated.T @ Vt_truncated
    subspace_error = float(np.linalg.norm(
        projection_full - projection_truncated, ord="fro"))
    singular_error = float(np.max(
        np.abs(singular_truncated - singular_full[:k]) /
        (np.abs(singular_full[:k]) + 1e-15)))

    connector = MobiusConnector(D, d_prime=k, mode="pca", seed=0).fit(matrix)
    connector_projection = connector.C @ connector.C.T
    true_projection = frame @ frame.T
    connector_error = float(np.linalg.norm(
        true_projection - connector_projection, ord="fro"))

    target_frame, _ = qr_twice(rng.standard_normal((300, k)))
    target = latent @ target_frame.T + 1e-9 * rng.standard_normal((N, 300))
    bridge = StateBridge(max_dense_elements=1, max_anchor_rank=k).fit(matrix, target)
    bridge_error = float(np.linalg.norm(bridge.align(matrix) - target) /
                         np.linalg.norm(target))
    print(f"  [Test N] randomized SVD k={k}: subspace_err={subspace_error:.2e} "
          f"S_err={singular_error:.2e} connector_err={connector_error:.2e} "
          f"bridge_err={bridge_error:.2e}")
    return (subspace_error < 1e-6 and singular_error < 1e-6
            and connector_error < 1e-6 and bridge_error < 1e-6
            and bridge.svd_truncated)


def test_groundtruth_bridge():
    """Bridge debe recuperar M_true cuando Y = X·M + ruido pequeño."""
    rng = np.random.default_rng(42)
    N, dA, dB = 400, 16, 24
    X = rng.standard_normal((N, dA))
    # QR de una matriz alta para construir una isometría rectangular dA×dB
    # con filas ortonormales. QR reducido sobre (dA,dB) no produce esa forma.
    frame, _ = np.linalg.qr(rng.standard_normal((dB, dA)), mode="reduced")
    M_true = frame.T
    Y = X @ M_true + 0.01 * rng.standard_normal((N, dB))
    b = StateBridge().fit(X, Y)
    err = np.linalg.norm(X @ b.W + b.bias - Y) / np.linalg.norm(Y)

    # Mismo control en dimensión alta: la representación compacta debe
    # recuperar el mapa sobre el subespacio observado sin crear W dA×dB.
    latent = rng.standard_normal((32, 8))
    frame_A, _ = qr_twice(rng.standard_normal((1600, 8)))
    frame_B, _ = qr_twice(rng.standard_normal((1500, 8)))
    X_large, Y_large = latent @ frame_A.T, latent @ frame_B.T
    compact = StateBridge().fit(X_large, Y_large)
    compact_pred = np.asarray([compact.align(row) for row in X_large])
    compact_err = np.linalg.norm(compact_pred - Y_large) / np.linalg.norm(Y_large)
    print(f"  [Test A] dense ground truth err = {err:.4f}  (esperado < 0.05)")
    print(f"  [Test A] compact bridge err = {compact_err:.2e}  "
          f"representation={compact.representation}, rank={compact.effective_rank}")
    return err < 0.05 and compact.representation in {
        "anchor_low_rank_procrustes", "anchor_randomized_low_rank_procrustes"} \
        and compact.W is None and compact_err < 1e-8


def test_frequency_warp():
    """
    Procrustes en log-dominio DEBE recuperar Y = |X|^{T_B/T_A}.
    Procrustes en z-dominio NO puede (err ~ 0.5).
    """
    rng = np.random.default_rng(42)
    N, d = 200, 128
    X = rng.standard_normal((N, d))
    T_A, T_B = 1/5, 1/7
    ratio = T_B / T_A
    Y = np.sign(X) * np.abs(X) ** ratio

    # z-dominio
    bz = StateBridge().fit(X, Y)
    err_z = np.linalg.norm(X @ bz.W + bz.bias - Y) / np.linalg.norm(Y)

    # log-dominio
    bl = LogStateBridge().fit(X, Y)
    Y_log_pred = np.array([bl.align(X[i]) for i in range(N)])
    err_log = np.linalg.norm(Y_log_pred - Y) / np.linalg.norm(Y)

    print(f"  [Test B] z-dominio err   = {err_z:.4f}  (referencia observada)")
    print(f"  [Test B] log-dominio err = {err_log:.6f}  (esperado < 1e-6)")
    print(f"  [Test B] scale_ratio empírico = {bl.scale_ratio:.6f}  "
          f"(esperado {ratio:.6f})")
    return err_log < 1e-6 and abs(bl.scale_ratio - ratio) < 1e-4


def test_shared_basis():
    """Control sintético: dos marcos ortonormales del mismo subespacio latente."""
    rng = np.random.default_rng(21)
    alphabet = list("بتثجحخدذرزسشصضطظعغ")
    corpus = ["".join(rng.choice(alphabet, size=8)) for _ in range(96)]
    trigrams = set()
    for t in corpus:
        tn = ar_norm(t)
        for i in range(max(0, len(tn)-2)):
            trigrams.add(tn[i:i+3])
    trigrams = sorted(trigrams)

    eA = SharedBasisEmbedder(trigrams, dim=24, seed=1, basis_dim=12)
    eB = SharedBasisEmbedder(trigrams, dim=32, seed=2, basis_dim=12)
    X = np.array([eA(t) for t in corpus])
    Y = np.array([eB(t) for t in corpus])

    b = StateBridge().fit(X, Y)
    err = np.linalg.norm(X @ b.W + b.bias - Y) / np.linalg.norm(Y)
    diag = diagnose_alignment(X, Y)
    print(f"  [Test C] shared-basis err = {err:.4f}  (esperado < 0.05)")
    print(f"  [Test C] ratio vs random  = {diag['ratio']:.3f}  {diag['verdict']}")
    return err < 0.05


def test_hash_no_signal():
    """Hashes independientes: control negativo evaluado fuera de muestra."""
    rng = np.random.default_rng(10)
    alphabet = list("بتثجحخدذرزسشصضطظعغ")
    corpus = ["".join(rng.choice(alphabet, size=12)) for _ in range(100)]
    eA = TrigramEmbedder(dim=32, salt=1)
    eB = TrigramEmbedder(dim=48, salt=2)
    X = np.array([eA(t) for t in corpus])
    Y = np.array([eB(t) for t in corpus])
    diag = diagnose_alignment(X, Y)
    print(f"  [Test D] hash-salt validation err = {diag['err_real']:.4f}  "
          f"ratio = {diag['ratio']:.3f}  {diag['verdict']}")
    return diag["ratio"] > 0.8


def test_z_laplacian():
    """Verifica identidad 1/7 + 4/(7·ñ) para ñ=10."""
    v = verify_z_laplacian(1/5, 10)
    print(f"  [Test E] T=1/5, ñ=10: consistent={v['consistent']}, "
          f"ttl_natural={v['ttl_natural']}")
    return v["consistent"] and v["ttl_natural"] == 5


def test_udp_projection():
    """Proyecta un Gram pequeño y audita normas y aristas tras factorizar."""
    base = TrigramEmbedder(dim=8, salt=3)
    corpus = ["aabbcc", "bbccdd", "ccddaa", "ddaabb"]
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    udp = UDPEmbedder(base, edges, output_dim=4, n_iter=1000, tol=1e-9).fit(corpus)
    report = udp.report
    print(f"  [Test F] UDP: converged={report['converged']} "
          f"edge_error={report['max_edge_error']:.2e} "
          f"Gram_distortion={report['relative_gram_distortion']:.4f}")
    return bool(report["accepted"] and report["max_edge_error"] < 1e-6)


def test_geometric_invariants():
    """Comprueba ortogonalidad, levantamiento y retorno de las seis fases."""
    rng = np.random.default_rng(8)
    frame, _ = qr_twice(rng.standard_normal((12, 6)))
    connector = MobiusConnector(12, d_prime=6, mode="random", seed=4).fit(
        rng.standard_normal((20, 12)))
    v = rng.standard_normal(12)
    R = connector.rotor_content(v)
    lifted = connector.lift(R)
    fast_lifted = connector.apply_lift(R, v)
    averaged, phase_std = FractalAverager(connector).apply(v)
    ok = (np.allclose(frame.T @ frame, np.eye(6), atol=1e-10)
          and ortho_drift(R) < 1e-10
          and ortho_drift(lifted) < 1e-10
          and np.allclose(np.linalg.norm(lifted @ v), np.linalg.norm(v), atol=1e-10)
          and np.allclose(fast_lifted, lifted @ v, atol=1e-10)
          and np.array_equal(averaged, v)
          and np.array_equal(phase_std, np.zeros_like(v)))
    print(f"  [Test G] geometría: orth(R)={ortho_drift(R):.2e} "
          f"orth(T)={ortho_drift(lifted):.2e} "
          f"fast_lift_error={np.linalg.norm(fast_lifted-lifted@v):.2e} "
          f"phase_error={np.linalg.norm(averaged-v):.2e}")
    return bool(ok)


def test_rag_and_sqlite(db_path: Optional[str] = None):
    """El RAG recupera un ancla procesada y SQLite restaura Task/checkpoint."""
    corpus = ["لا حيلة إلا الله", "بسم الله الرحمن الرحيم",
              "قل هو الله أحد", "الحمد لله رب العالمين"]
    references = ["demo:1", "demo:2", "demo:3", "demo:4"]
    cfg = A2AConfig(dim=32, d_prime=6, connector_mode="hybrid",
                    n_tilde=10, ttl=3, use_fractal_avg=True, seed=11)
    A = Agent("RAG-A", TrigramEmbedder(32, salt=5), cfg, dim=32)
    B = Agent("RAG-B", TrigramEmbedder(32, salt=5), cfg, dim=32)
    rag = RAGDecodeInterface(top_k=4, confidence_threshold=0.999)
    temporary_directory = tempfile.TemporaryDirectory() if db_path is None else None
    if temporary_directory is not None:
        db_path = str(Path(temporary_directory.name) / "tasks.sqlite3")
    elif Path(db_path).exists():
        raise FileExistsError(f"El test se niega a reutilizar una base existente: {db_path}")
    try:
        store = SQLiteTaskStore(str(db_path))
        proto = Protocol(A, B, decoder=rag, store=store).calibrate(corpus, references)
        try:
            task = proto.send(corpus[0], turns=1)
            decoded = task.artifacts["decoded"]
            restored = store.load(task.task_id)
            checkpoint = store.load_checkpoint(task.task_id)
            pending = Task(task_id="interrupted-demo", state=TaskState.WORKING)
            pending.add_turn(np.ones(3), {"role": "checkpoint"})
            store.save(pending)
            recovered = store.recover_interrupted()
            recovered_task = store.load(pending.task_id)
            ok = (decoded["accepted_from_corpus"]
                  and decoded["text"] == corpus[0]
                  and decoded["reference"] == references[0]
                  and restored is not None and restored.state == TaskState.COMPLETED
                  and checkpoint is not None
                  and np.allclose(checkpoint, task.history[-1].latent)
                  and recovered == 1 and recovered_task.state == TaskState.FAILED)
            print(f"  [Test H] RAG={decoded['reference']} score={decoded['score']:.6f}; "
                  f"SQLite turns={len(restored.history)} recovered={recovered}")
            return bool(ok)
        finally:
            proto.close()
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def test_secure_rag_scale():
    """El hot path verifica filas candidatas, no recorre todo el índice."""
    rng = np.random.default_rng(0)
    n_rows, dim = 500, 32
    embeddings = rng.standard_normal((n_rows, dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    texts = [f"versiculo {index}" for index in range(n_rows)]
    references = [f"v:{index}" for index in range(n_rows)]
    rag = RAGDecodeInterface(embeddings, texts, corpus_ids=references,
                             top_k=3, confidence_threshold=0.5)
    secure = SecureRAGIndex(rag, b"k" * 32)

    # La auditoría explícita detecta cambios al objeto fuente y permite
    # recuperarse si se restaura el índice firmado.
    original_text = rag.corpus_texts[0]
    rag.corpus_texts[0] = "texto alterado"
    tamper_detected = not secure.verify()
    rag.corpus_texts[0] = original_text
    restored = secure.verify()

    # Decode no debe volver a ejecutar la auditoría O(N) del índice completo.
    def forbid_full_scan():
        raise AssertionError("decode() ejecutó la verificación completa del índice")
    secure._verify_index = forbid_full_scan
    queries = embeddings[:20]
    started = time.perf_counter()
    results = [secure.decode(query) for query in queries]
    elapsed = time.perf_counter() - started
    per_query_ms = 1000.0 * elapsed / len(queries)
    valid = all(item["corpus_signature_verified"] and
                item["accepted_from_corpus"] and
                item["reference"] == references[index]
                for index, item in enumerate(results))
    print(f"  [Test K] SecureRAG N={n_rows}: {per_query_ms:.3f} ms/query; "
          f"auditoría completa fuera del hot path={tamper_detected and restored}")
    return bool(tamper_detected and restored and valid)


def test_fractal_noise_invariant():
    """La fórmula optimizada conserva el resultado del levantamiento original."""
    rng = np.random.default_rng(3)
    connector = MobiusConnector(64, d_prime=6, mode="random", seed=1).fit(
        rng.standard_normal((20, 64)))
    averager = FractalAverager(connector)
    vector = rng.standard_normal(64)
    noise_std, seed = 1e-3, 42
    mean, std = averager.apply(vector, noise_std=noise_std, seed=seed)

    reference_rng = np.random.default_rng(seed)
    reference = []
    for index in range(6):
        phase = averager._phase_rotation(index * np.pi / 3)
        sample = connector.apply_lift(phase, vector)
        sample += reference_rng.normal(0.0, noise_std, size=vector.shape)
        reference.append(connector.apply_lift(phase.T, sample))
    reference = np.stack(reference)
    ratio = np.linalg.norm(mean) / np.linalg.norm(vector)
    equivalent = (np.allclose(mean, reference.mean(axis=0), atol=1e-12, rtol=1e-12)
                  and np.allclose(std, reference.std(axis=0), atol=1e-12, rtol=1e-12))
    print(f"  [Test L] fractal ruido: ||mean||/||v||={ratio:.6f}; "
          f"equivalencia={equivalent}")
    return bool(equivalent and 0.95 < ratio < 1.05)


def test_ar_norm_equivalence():
    """str.translate conserva la normalización árabe anterior."""
    def old_ar_norm(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        marks = (set(range(0x0610, 0x061B)) | set(range(0x064B, 0x0660))
                 | set(range(0x06D6, 0x06EE)))
        text = "".join(char for char in text if ord(char) not in marks)
        for source, target in (("إ", "ا"), ("أ", "ا"), ("آ", "ا"), ("ٱ", "ا"),
                               ("ى", "ي"), ("ة", "ه")):
            text = text.replace(source, target)
        return " ".join(text.split())

    cases = ["لَا حِيلَةَ إِلَّا اللَّه", "إبراهيم", "شيء", "مُحَمَّد",
             "ٱلرَّحْمَٰنُ", "  الحمد   لله  رب العالمين  "]
    equivalent = all(old_ar_norm(text) == ar_norm(text) for text in cases)
    print(f"  [Test M] ar_norm equivalencia={equivalent}")
    return equivalent


def test_end_to_end_local():
    """Protocolo completo en transporte local."""
    corpus = ["لا حيلة إلا الله", "بسم الله الرحمن الرحيم",
              "قل هو الله أحد", "الحمد لله رب العالمين",
              "الرحمن الرحيم", "مالك يوم الدين"]
    cfg = A2AConfig(dim=128, d_prime=6, connector_mode="hybrid",
                     n_tilde=10, ttl=5, use_fractal_avg=True)
    A = Agent("A", TrigramEmbedder(128, salt=0), cfg, dim=128)
    B = Agent("B", TrigramEmbedder(256, salt=0), cfg, dim=256)   # mismo salt para señal
    proto = Protocol(A, B).calibrate(corpus)

    task = proto.send("لا حيلة إلا الله", turns=2)
    ok = (task.state == TaskState.COMPLETED and len(task.history) >= 3)
    print(f"  [Test I] local: state={task.state.value} turns={len(task.history)}  "
          f"bridge_level={proto.bridge_meta['level']}  "
          f"validation_err={proto.bridge_meta.get('validation_error_nivel1', 'n/a')}")
    print(f"           decoded hash: {task.artifacts['decoded']['hash_original']}")
    return ok


def test_end_to_end_http():
    """Protocolo completo en transporte HTTP JSON-RPC."""
    corpus = ["لا حيلة إلا الله", "بسم الله الرحمن الرحيم",
              "قل هو الله أحد", "الحمد لله رب العالمين",
              "الرحمن الرحيم", "مالك يوم الدين"]
    cfg = A2AConfig(dim=128, d_prime=6, connector_mode="hybrid",
                     n_tilde=10, ttl=5, use_fractal_avg=False)
    A = Agent("A", TrigramEmbedder(128, salt=0), cfg, dim=128)
    B = Agent("B", TrigramEmbedder(256, salt=0), cfg, dim=256)
    proto = Protocol(A, B, transport="http",
                     host_A="127.0.0.1", port_A=0,
                     host_B="127.0.0.1", port_B=0).calibrate(corpus)
    try:
        manifest = proto.client_A.discover()
        print(f"  [Test J] discovery: name={manifest['name']} "
              f"dims={manifest['capabilities']['latent_dims']}")
        out = proto.send("لا حيلة إلا الله", turns=2)
        ok = out["state"] == "completed" and "decoded" in out
        print(f"  [Test J] HTTP JSON-RPC: state={out['state']} turns={out['turns_http']}  "
              f"bridge_level={out['bridge_meta']['level']}")
        print(f"           decoded hash: {out['decoded']['hash_original']}")
        return ok
    finally:
        proto.close()


# ═══════════════════════════════════════════════════════════════════════════
# 18. MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print(" a2a_mobius.py — SUITE DE TESTS")
    print("═" * 70)

    print("\n── Test A: bridge ground-truth ──")
    ok_A = test_groundtruth_bridge()

    print("\n── Test B: warp de frecuencia (T_A ≠ T_B) ──")
    ok_B = test_frequency_warp()

    print("\n── Test C: base compartida (señal lineal) ──")
    ok_C = test_shared_basis()

    print("\n── Test D: hash-salt distinto (control negativo) ──")
    ok_D = test_hash_no_signal()

    print("\n── Test E: consistencia z-Laplace ──")
    ok_E = test_z_laplacian()

    print("\n── Test F: proyección UDP con auditoría ──")
    ok_F = test_udp_projection()

    print("\n── Test G: invariantes geométricos ──")
    ok_G = test_geometric_invariants()

    print("\n── Test H: RAG + persistencia SQLite ──")
    ok_H = test_rag_and_sqlite(os.environ.get("A2A_SQLITE_TEST_DB"))

    print("\n── Test K: hot path de SecureRAG ──")
    ok_K = test_secure_rag_scale()

    print("\n── Test L: invariantes con ruido fractal ──")
    ok_L = test_fractal_noise_invariant()

    print("\n── Test M: equivalencia de normalización árabe ──")
    ok_M = test_ar_norm_equivalence()

    print("\n── Test N: equivalencia de SVD aleatoria ──")
    ok_N = test_randomized_svd_equivalence()

    print("\n── Test I: protocolo end-to-end (local) ──")
    ok_I = test_end_to_end_local()

    print("\n── Test J: protocolo end-to-end (HTTP JSON-RPC) ──")
    ok_J = test_end_to_end_http()

    print("\n" + "═" * 70)
    checks = [ok_A, ok_B, ok_C, ok_D, ok_E, ok_F, ok_G, ok_H,
              ok_K, ok_L, ok_M, ok_N, ok_I, ok_J]
    print(f" RESUMEN: {sum(checks)}/{len(checks)} comprobaciones OK")
    print("═" * 70)
    print("""
 Alcance y límites:
 • A, B, C y G son casos sintéticos con relaciones conocidas, no pruebas semánticas.
 • D es un control negativo fuera de muestra; el ratio es una heurística.
 • E verifica una identidad aritmética, no una ley física.
 • F proyecta un Gram pequeño y reporta convergencia y distorsión.
 • H recupera un elemento exacto de un índice alineado y comprueba SQLite.
 • I/J comprueban integración local y HTTP de prototipo, no producción ni A2A.
 • Vec2Text es opcional y requiere corrector compatible e inversor al espacio nativo.
    """)
