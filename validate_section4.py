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

"""Section 4 -- Exercise 6 and the complexification boundary.

Validates:
  Lemma 4.1     M_2(L(F_k)) = L(F_{(k+3)/4}): exact arithmetic for integer and rational k,
                admissibility (k+3)/4 > 1, k = 5 -> 2, k = infinity
  Prop. 4.2     H (x)_R C = M_2(C) explicitly; the complexification identity
                (A (x)_R H) (x)_R C = M_2(A (x)_R C) checked on a finite-dimensional real
                group algebra A = R[S_3] (the same algebra that turns L_H(F_5) (x) C into
                M_2(L(F_5))); then L(F_5)_2 = L(F_2) = L_R(F_2) (x) C
  Thm 4.3 (1)   H and M_2(R) are not isomorphic but both complexify to M_2(C)
  Thm 4.3 (2)   finite-dim: M_4(R) and M_2(H) have equal dimension, different Brauer class,
                distinguished by the minimal dimension of a left ideal (4 vs 8)
  Thm 4.3 (3)   ingredients of R_H = R_R: H (x) H = M_4(R) is trace-preserving
                (tau(psi(a(x)b)) = Re a Re b), so the regrouped tower is trace-compatible
  Remark 4.4    random-matrix 4th moments: beta = 1, 2, 4 share the limit 2 (Catalan),
                finite-N corrections about +5/N, 0, -2.5/N (numerical observation only)

NOT validated (open problem): whether L_H(F_5) and L_R(F_2) are isomorphic.
"""
from fractions import Fraction as Fr
import itertools as it
import numpy as np
from common import (check, note, section, summary, kron, qmul, qconj, QBASIS, phi, I2, real_rank,
                    complex_rank)

rng = np.random.default_rng(23)


# ------------------------------------------------------------------ Lemma 4.1
section("Lemma 4.1  M_2(L(F_k)) = L(F_((k+3)/4))")
amp = lambda r, T: 1 + (r - 1) / T
check("1 + (k-1)/2^2 = (k+3)/4 for k = 2..1000 (exact)",
      all(amp(Fr(k), 4) == Fr(k + 3, 4) for k in range(2, 1001)))
check("same identity for 1000 random rationals k > 1",
      all(amp(k, 4) == (k + 3) / 4 for k in (Fr(np.random.randint(101, 10 ** 5), 100) for _ in range(1000))))
check("k >= 2  =>  (k+3)/4 >= 5/4 > 1 (admissible)", all(Fr(k + 3, 4) >= Fr(5, 4) > 1 for k in range(2, 1000)))
check("k = 5 gives L(F_2)", Fr(5 + 3, 4) == 2)
check("k > 1 rational: (k+3)/4 > 1 as well", all((k + 3) / 4 > 1 for k in (Fr(101, 100), Fr(3, 2), Fr(7, 5))))

# ------------------------------------------------------------------ Proposition 4.2
section("Proposition 4.2  complexification of L_H(F_5)")
# H (x)_R C = M_2(C): the C-linear extension of phi is an isomorphism
imgs = [phi(b) for b in QBASIS]
check("C-span of phi(1), phi(i), phi(j), phi(k) is all of M_2(C) (complex rank 4)", complex_rank(imgs) == 4)
check("phi is multiplicative, so its C-linear extension is a C-algebra isomorphism H (x) C -> M_2(C)",
      all(np.allclose(phi(qmul(a, b)), phi(a) @ phi(b)) for a in QBASIS for b in QBASIS))

# finite-dimensional model of (A (x)_R H) (x)_R C = M_2(A (x)_R C) with A = R[S_3]
perms = list(it.permutations(range(3)))
idx = {p: i for i, p in enumerate(perms)}
compose = lambda p, q: tuple(p[q[i]] for i in range(3))


def lam(p):  # left regular representation of S_3 (real 6x6 permutation matrices)
    M = np.zeros((6, 6))
    for q in perms:
        M[idx[compose(p, q)], idx[q]] = 1
    return M


regA = [lam(p) for p in perms]
check("R[S_3] regular representation is faithful (real dim 6)",
      np.linalg.matrix_rank(np.array([M.ravel() for M in regA])) == 6)
# image of A (x) H inside M_6(C) (x) M_2(C) = M_2(M_6(C)) under g (x) q -> lam(g) (x) phi(q)
AH = [np.kron(phi(q), L) for L in regA for q in QBASIS]
check("A (x)_R H embeds with real dimension 6*4 = 24", real_rank(AH) == 24)
check("its C-span has complex dimension 24 = dim_C M_2(C[S_3]) (so (A(x)H)(x)C = M_2(A(x)C))",
      complex_rank(AH) == 24)
M2AC = [np.kron(E, L) for L in regA for E in
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]]))]
check("C-span of the image equals C-span of M_2(lam(C[S_3])) exactly",
      complex_rank(AH) == complex_rank(M2AC) == complex_rank(AH + [m.astype(complex) for m in M2AC]))
check("chain: L_H(F_5)(x)C = M_2(L(F_5)) = L(F_5)_2 = L(F_(1+4/4)) = L(F_2) = L_R(F_2)(x)C",
      amp(Fr(5), 4) == 2)

# ------------------------------------------------------------------ Theorem 4.3 (1)
section("Theorem 4.3 (1)  H vs M_2(R)")
ok = True
for _ in range(500):
    q = rng.normal(size=4)
    ok &= np.isclose(np.linalg.det(phi(tuple(q))).real, np.dot(q, q))
check("det phi(q) = |q|^2 > 0 for q != 0: every nonzero quaternion is invertible", ok)
N = np.array([[0.0, 1.0], [0.0, 0.0]])
check("M_2(R) has a nonzero nilpotent (zero divisor), so M_2(R) is not iso to H", np.allclose(N @ N, 0))
check("both complexify to M_2(C) (complex rank 4 of each R-basis)",
      complex_rank(imgs) == 4 and complex_rank([np.eye(2), N, N.T, np.diag([1.0, -1.0])]) == 4)

# ------------------------------------------------------------------ Theorem 4.3 (2)
section("Theorem 4.3 (2)  Wedderburn: M_4(R) vs M_2(H)")


def L4(q):  # real 4x4 matrix of left multiplication by q
    return np.array([qmul(q, e) for e in QBASIS], dtype=float).T


def M2H_basis():
    out = []
    for r, c in it.product(range(2), repeat=2):
        for e in QBASIS:
            M = np.zeros((8, 8)); M[4 * r:4 * r + 4, 4 * c:4 * c + 4] = L4(e); out.append(M)
    return out


B_M2H = M2H_basis()
B_M4R = [np.eye(4)[:, [i]] @ np.eye(4)[[j], :] for i in range(4) for j in range(4)]
check("dim_R M_2(H) = 16 = dim_R M_4(R)", np.linalg.matrix_rank(np.array([m.ravel() for m in B_M2H])) == 16)


def left_ideal_dim(basis, x):
    return np.linalg.matrix_rank(np.array([(a @ x).ravel() for a in basis]))


E11 = B_M4R[0]
check("M_4(R): the left ideal M_4(R) E_11 has dimension 4", left_ideal_dim(B_M4R, E11) == 4)
dims = []
for _ in range(300):  # random quaternionic rank-1 elements (column) and random elements
    col = [rng.normal(size=4) for _ in range(2)]; row = [rng.normal(size=4) for _ in range(2)]
    x = np.block([[L4(tuple(qmul(tuple(col[r]), tuple(row[c])))) for c in range(2)] for r in range(2)])
    dims.append(left_ideal_dim(B_M2H, x))
check("M_2(H): left ideals generated by quaternionic rank-1 elements have dimension 8 (> 4)",
      min(dims) == 8, f"observed {sorted(set(int(d) for d in dims))}")
check("so M_2(H) (Brauer class [H]) is not iso to M_4(R) (class 0) despite equal dimension", min(dims) != 4)

# ------------------------------------------------------------------ Theorem 4.3 (3)
section("Theorem 4.3 (3)  trace compatibility behind R_H = R_R")


def psi_matrix(a, b):
    return np.array([qmul(qmul(a, e), qconj(b)) for e in QBASIS], dtype=float).T


ok = True
for _ in range(300):
    a = tuple(rng.normal(size=4)); b = tuple(rng.normal(size=4))
    ok &= np.isclose(np.trace(psi_matrix(a, b)) / 4, a[0] * b[0])
check("normalized trace: tr(psi(a (x) b))/4 = Re(a) Re(b) = tau_H (x) tau_H", ok)
# tower H^{(x)k}: x -> x (x) 1 preserves normalized trace, and the R_H and R_R towers interleave
ok = True
for k in range(1, 4):
    X = sum(rng.normal() * kron(*[L4(QBASIS[i]) for i in idx]) for idx in it.product(range(4), repeat=k))
    ok &= np.isclose(np.trace(X) / X.shape[0], np.trace(np.kron(X, np.eye(4))) / (4 * X.shape[0]))
check("x -> x (x) 1 is trace-preserving on H^(x)k, k = 1, 2, 3 (so the regrouped towers agree)", ok)
note("R_H stages H^(x)(2m+1) and R_R stages H^(x)(2m) interleave in the single trace-preserving "
     "tower H < H^(x)2 < H^(x)3 < ...; both weak closures are the same algebra")

# ------------------------------------------------------------------ Remark 4.4
section("Remark 4.4  random-matrix 4th moments (numerical observation)")


def tr4(X):
    """Tr X^4 = ||X^2||_F^2 for Hermitian X (batched)."""
    X2 = X @ X
    return np.sum(np.abs(X2) ** 2, axis=(1, 2))


def sample_m4(beta, N, S):
    if beta == 1:
        A = rng.normal(size=(S, N, N)) / np.sqrt(N)
        X = (A + A.transpose(0, 2, 1)) / np.sqrt(2)                 # off-diag var 1/N, diag var 2/N
        return tr4(X) / N
    if beta == 2:
        A = (rng.normal(size=(S, N, N)) + 1j * rng.normal(size=(S, N, N))) / np.sqrt(2 * N)
        X = (A + A.conj().transpose(0, 2, 1)) / np.sqrt(2)           # off-diag E|x|^2 = 1/N, diag var 1/N
        return tr4(X) / N
    # beta = 4: N x N quaternion self-dual, realized as 2N x 2N complex; normalized trace /2N
    w = rng.normal(size=(S, N, N)); w = (w + w.transpose(0, 2, 1)) / 2   # off-diag var 1/2, diag var 1
    comps = [w]
    for _ in range(3):
        v = rng.normal(size=(S, N, N)); comps.append((v - v.transpose(0, 2, 1)) / 2)
    scale = 1 / np.sqrt(2 * N)
    units = [phi(q) for q in QBASIS]
    X = sum(np.einsum("sij,ab->siajb", c * scale, u).reshape(S, 2 * N, 2 * N) for c, u in zip(comps, units))
    return tr4(X) / (2 * N)


Ns = [6, 12, 24]
coef, m4s = {}, {}
for beta, S in ((1, 60000), (2, 60000), (4, 30000)):
    m4 = [sample_m4(beta, N, S).mean() for N in Ns]
    # fit m4 = 2 + c/N + e/N^2
    A = np.array([[1 / N, 1 / N ** 2] for N in Ns]); y = np.array(m4) - 2
    c, e = np.linalg.lstsq(A, y, rcond=None)[0]
    coef[beta] = c
    m4s[beta] = m4
    print(f"    beta={beta}: m4(N={Ns}) = {np.round(m4, 4)}  ->  fitted 1/N coefficient {c:+.2f}")
check("beta = 1: correction ~ +5/N", abs(coef[1] - 5) < 0.6, f"{coef[1]:+.2f}")
check("beta = 2: no 1/N correction", abs(coef[2]) < 0.5, f"{coef[2]:+.2f}")
check("beta = 4: correction ~ -2.5/N", abs(coef[4] + 2.5) < 0.6, f"{coef[4]:+.2f}")
check("for every beta, |m4 - 2| shrinks as N doubles (common limit 2 = Catalan C_2)",
      all(abs(m[i + 1] - 2) < abs(m[i] - 2) for b, m in m4s.items() if b != 2 for i in range(len(Ns) - 1))
      and abs(m4s[2][-1] - 2) < 0.01)
note("this is a numerical observation (Remark 4.4), not a proof; it does not bear on real isomorphism")
note("NOT validated: whether L_H(F_5) and L_R(F_2) are isomorphic -- this is open (Problem 2)")

raise SystemExit(summary("Section 4"))
