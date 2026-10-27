# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Validation code for: "Towards a Classification of the Free Group Factors L_K(F_k)
# over R, C and H as Real von Neumann Algebras" (Cooperman & Petrunina V.V., 27 October 2026).

"""Section 5 -- beta-microstate free entropy dimension (corrected statement).

Conjecture 5.1 is open; no computation proves or refutes it. This script validates the
concrete mathematical statements Section 5 makes around it:

  d_beta(N)        real dimension of N x N self-adjoint matrices over R, C, H equals
                   N + beta N(N-1)/2  (computed from explicit bases)
  draft dims       the draft's dim o(N), dim u(N), dim sp(N/2) are Lie-algebra dimensions and
                   differ from the microstate-space dimensions d_beta(N)
  volume estimate  log-volumes of comparable balls in R^{d_1} and R^{d_2} differ at order
                   N^2 log N, so the draft's O(1/N) estimate is false
  counterexample   (S1, S2, S1): the moment conditions force ||Y3 - Y1||_2^2 <= 4 eps, so the
                   repeated coordinate adds no microstate dimension (delta_0 = 2 < 3 = k)
  asymptotic       independent GOE / GUE / GSE matrices are asymptotically free:
  freeness         tr(ABAB) -> 0 and tr(A^2 B^2) -> 1 for beta = 1, 2, 4
"""
import itertools as it
import math
import numpy as np
from common import check, note, section, summary, phi, QBASIS, real_rank

rng = np.random.default_rng(31)


# ------------------------------------------------------------------ d_beta(N)
section("d_beta(N) = N + beta N(N-1)/2")


def selfadjoint_basis(beta, N):
    """Spanning set of N x N self-adjoint matrices over R / C / H, realized as complex matrices
    (H entries as 2x2 blocks via phi)."""
    out = []
    units = {1: [1.0], 2: [1.0, 1j], 4: [phi(q) for q in QBASIS]}[beta]
    blk = 2 if beta == 4 else 1
    for i in range(N):
        for j in range(N):
            for u in units:
                E = np.zeros((N, N)); E[i, j] = 1
                M = np.kron(E, u) if beta == 4 else E * u
                out.append(M + M.conj().T)      # self-adjoint part
    return out


ok = True
for N in range(1, 6):
    for beta in (1, 2, 4):
        d = real_rank(selfadjoint_basis(beta, N))
        ok_here = d == N + beta * N * (N - 1) // 2
        ok &= ok_here
        if not ok_here:
            print("   mismatch", beta, N, d)
check("explicit bases: dim_R = N + beta N(N-1)/2 for beta = 1, 2, 4 and N = 1..5", ok)
check("closed forms: d_1 = N(N+1)/2, d_2 = N^2, d_4 = N(2N-1)",
      all(N + N * (N - 1) // 2 == N * (N + 1) // 2 and N + 2 * N * (N - 1) // 2 == N * N
          and N + 4 * N * (N - 1) // 2 == N * (2 * N - 1) for N in range(1, 200)))

section("draft dimension list is a different object")
for N in (4, 10, 100):
    lie = (N * (N - 1) // 2, N * N, (N // 2) * (2 * (N // 2) + 1))
    sa = (N * (N + 1) // 2, N * N, N * (2 * N - 1))
    check(f"N={N}: Lie dims (o, u, sp) = {lie} differ from self-adjoint dims d_beta = {sa}",
          lie[0] != sa[0] and lie[2] != sa[2])

# ------------------------------------------------------------------ volume estimate
section("draft estimate |log vol Gamma_beta - log vol Gamma_2| = O(1/N) is false")


def log_ball(d, R):
    """log volume of the Euclidean ball of radius R in R^d."""
    return d / 2 * math.log(math.pi) - math.lgamma(d / 2 + 1) + d * math.log(R)


rows = []
for N in (10, 100, 1000, 10000):
    R = math.sqrt(N)                          # tr-normalized unit ball: ||X||_HS <= sqrt(N)
    diff = abs(log_ball(N * (N + 1) // 2, R) - log_ball(N * N, R))
    rows.append((N, diff, diff / (N * N * math.log(N))))
    print(f"    N={N:>6}: |log vol_1 - log vol_2| = {diff:.3e},  ratio to N^2 log N = {diff / (N * N * math.log(N)):.3f}")
check("the difference grows (it is not O(1/N))", all(rows[i + 1][1] > rows[i][1] for i in range(len(rows) - 1)))
check("it is of order N^2 log N (ratio bounded away from 0)", rows[-1][2] > 0.05)

# ------------------------------------------------------------------ counterexample (S1, S2, S1)
section("counterexample to 'delta = k for every generating tuple'")
N, eps = 200, 1e-3
ok = True
for _ in range(50):
    Y1 = rng.normal(size=(N, N)); Y1 = (Y1 + Y1.T) / np.sqrt(2 * N)
    Y3 = Y1 + rng.normal(scale=rng.uniform(0, 0.05), size=(N, N)); Y3 = (Y3 + Y3.T) / 2
    tr = lambda A: np.trace(A) / N
    m11, m33, m13 = tr(Y1 @ Y1), tr(Y3 @ Y3), tr(Y1 @ Y3)
    ok &= np.isclose(tr((Y1 - Y3) @ (Y1 - Y3)), m11 + m33 - 2 * m13)
check("identity tr((Y1-Y3)^2) = m11 + m33 - 2 m13 (50 random cases)", ok)
worst = max((1 + a) + (1 + b) - 2 * (1 - c) for a, b, c in it.product((-eps, 0, eps), repeat=3))
check("moment errors <= eps (targets 1, 1, 1 for (S1, S2, S1)) force tr((Y1-Y3)^2) <= 4 eps",
      math.isclose(worst, 4 * eps), f"worst case {worst:.4g} = 4 eps")
note("so Y3 lies in a ball of radius 2 sqrt(eps) around Y1: the third coordinate contributes no "
     "dimension, and delta_0(S1, S2, S1) = delta_0(S1, S2) = 2 < 3 (Voiculescu); the draft's claim fails")

# ------------------------------------------------------------------ asymptotic freeness
section("asymptotic freeness of independent beta-ensembles (Gamma_beta non-empty)")


def sample(beta, N):
    if beta == 1:
        A = rng.normal(size=(N, N)); return (A + A.T) / np.sqrt(2 * N)
    if beta == 2:
        A = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2 * N)
        return (A + A.conj().T) / np.sqrt(2)
    w = rng.normal(size=(N, N)); w = (w + w.T) / 2
    comps = [w] + [(lambda v: (v - v.T) / 2)(rng.normal(size=(N, N))) for _ in range(3)]
    return sum(np.kron(c, phi(q)) for c, q in zip(comps, QBASIS)) / np.sqrt(2 * N)


for beta in (1, 2, 4):
    N = 300 if beta < 4 else 150
    A, B = sample(beta, N), sample(beta, N)
    tr = lambda M: np.real(np.trace(M)) / M.shape[0]
    abab, a2b2, a2 = tr(A @ B @ A @ B), tr(A @ A @ B @ B), tr(A @ A)
    check(f"beta={beta}: tr(A^2) ~ 1, tr(A^2 B^2) ~ 1, tr(ABAB) ~ 0 (free semicircular values)",
          abs(a2 - 1) < 0.05 and abs(a2b2 - 1) < 0.08 and abs(abab) < 0.05,
          f"{a2:.3f}, {a2b2:.3f}, {abab:+.3f}")
note("Conjecture 5.1 itself (equality of delta_0,beta for beta = 1, 2, 4) is open and is not tested")

raise SystemExit(summary("Section 5"))
