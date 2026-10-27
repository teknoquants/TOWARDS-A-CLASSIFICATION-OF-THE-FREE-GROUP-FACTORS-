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

"""Section 1 -- Q-structure and Brauer parity.

Validates:
  Lemma 1.1        J_n^2 = (-1)^n I
  Proposition 1.2  J_n P J_n^{-1} = (-1)^{w(P)} P;  iP in k_n  <=>  w(P) odd;
                   dim k_n = (4^n - (-2)^n)/2 = dim sp(2^{n-1}) (n odd) / dim so(2^n) (n even);
                   k_n is closed under brackets; the J_n-fixed subalgebra of su(2^n)
                   computed directly by linear algebra has the same dimension.
  phi              phi(Q) = commutant of J_1 in M_2(C), phi is an injective algebra map,
                   unit quaternions -> SU(2), and phi extends to H (x)_R C = M_2(C).
  Lemma 1.3        psi(a (x) b)(x) = a x conj(b) is an algebra isomorphism H (x) H -> M_4(R)
                   (checked in exact integer arithmetic on all 16 x 16 basis products).
  Corollary 1.4    C_n = phi(H)^{(x)n} = J_n-fixed real algebra (real dim 4^n);
                   rank-1 J_n-fixed projections exist iff n even (real type),
                   Kramers orthogonality <v, J_n v> = 0 for n odd (quaternionic type);
                   [H] has order exactly 2 (H is a division algebra, H (x) H = M_4(R)).
"""
import itertools as it
import numpy as np
from common import (check, note, section, summary, kron, SY, I2, SX, SZ, pauli_string,
                    all_pauli_labels, weight, J_apply_conj, J_squared, mask,
                    lie_closure_dim, qmul, qconj, QBASIS, qnorm2, phi,
                    real_rank, complex_rank)

rng = np.random.default_rng(7)

# ------------------------------------------------------------------ Lemma 1.1
section("Lemma 1.1  J_n^2 = (-1)^n I")
for n in range(1, 8):
    check(f"n={n}: J_n^2 = {(-1) ** n:+d} I",
          np.allclose(J_squared(n), (-1) ** n * np.eye(2 ** n)))

# ------------------------------------------------------------------ Proposition 1.2
section("Proposition 1.2  odd-weight criterion and dim k_n")
for n in range(1, 5):
    ok_sign = ok_member = True
    for lab in all_pauli_labels(n):
        if lab == "I" * n:
            continue
        P = pauli_string(lab)
        w = weight(lab)
        ok_sign &= np.allclose(J_apply_conj(n, P), (-1) ** w * P)
        member = np.allclose(J_apply_conj(n, 1j * P), 1j * P)
        ok_member &= (member == (w % 2 == 1))
    check(f"n={n}: J P J^-1 = (-1)^w(P) P for all {4 ** n - 1} Pauli strings", ok_sign)
    check(f"n={n}: iP is J-fixed  <=>  w(P) odd", ok_member)

for n in range(1, 10):
    count = sum(len(list(it.combinations(range(n), w))) * 3 ** w for w in range(1, n + 1, 2))
    formula = (4 ** n - (-2) ** n) // 2
    if n % 2:
        m = 2 ** (n - 1); lie = m * (2 * m + 1); name = f"sp({m})"
    else:
        m = 2 ** n; lie = m * (m - 1) // 2; name = f"so({m})"
    check(f"n={n}: #odd-weight strings = (4^n-(-2)^n)/2 = dim {name}",
          count == formula == lie, f"{count} = {formula} = {lie}")

# k_n closed under brackets (exact, via Pauli masks)
for n in range(1, 6):
    odd = [mask(l) for l in all_pauli_labels(n) if weight(l) % 2 == 1]
    d, _ = lie_closure_dim(odd)
    check(f"n={n}: span of odd-weight iP is a Lie algebra (closure adds nothing)",
          d == len(odd), f"dim {d}")

# Direct linear-algebra computation of {A in su(2^n): J A J^-1 = A}
for n in range(1, 4):
    N = 2 ** n
    basis = [1j * pauli_string(l) for l in all_pauli_labels(n) if l != "I" * n]  # su(N)
    # the map A -> J A J^-1 - A is R-linear; compute its kernel dimension on su(N)
    imgs = [J_apply_conj(n, A) - A for A in basis]
    rank = real_rank(imgs)
    kernel = len(basis) - rank
    check(f"n={n}: dim of J_n-fixed subalgebra of su({N}) computed directly",
          kernel == (4 ** n - (-2) ** n) // 2, f"{kernel}")

# Type of the symmetry group: symplectic (n odd) or orthogonal (n even).
# Write J = S K with S = sigma_y^{(x)n}. Fixed unitaries satisfy S conj(U) S^-1 = U,
# i.e. U^T (S) U = S when S^T = -S (symplectic form) or S^T = S (symmetric form).
for n in range(1, 6):
    S = kron(*[SY] * n)
    sym = np.allclose(S.T, S)
    anti = np.allclose(S.T, -S)
    expected = "symmetric (orthogonal type)" if n % 2 == 0 else "antisymmetric (symplectic type)"
    check(f"n={n}: sigma_y^(x)n is {expected}", sym if n % 2 == 0 else anti)
    # random fixed unitary: exp of a random element of k_n
    labs = [l for l in all_pauli_labels(n) if weight(l) % 2 == 1]
    A = sum(rng.normal() * 1j * pauli_string(l) for l in labs[:40])
    w, V = np.linalg.eigh(-1j * A)
    U = V @ np.diag(np.exp(1j * w)) @ V.conj().T
    check(f"n={n}: exp(k_n) preserves the bilinear form  U^T S U = S",
          np.allclose(U.T @ S @ U, S))

# ------------------------------------------------------------------ phi and SU(2)
section("phi: Q -> M_2(C)")
imgs = [phi(b) for b in QBASIS]
check("phi(H) is J_1-fixed", all(np.allclose(J_apply_conj(1, M), M) for M in imgs))
check("phi(H) has real dimension 4 = real dim of J_1-fixed algebra", real_rank(imgs) == 4)
ok = all(np.allclose(phi(qmul(a, b)), phi(a) @ phi(b)) for a in QBASIS for b in QBASIS)
check("phi is multiplicative on the basis (algebra homomorphism)", ok)
ok = True
for _ in range(200):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    U = phi(tuple(q))
    ok &= np.allclose(U.conj().T @ U, I2) and np.isclose(np.linalg.det(U), 1)
check("unit quaternions map into SU(2) (200 random samples)", ok)
check("phi(H) spans M_2(C) over C  (H (x)_R C = M_2(C))", complex_rank(imgs) == 4)

# ------------------------------------------------------------------ Lemma 1.3
section("Lemma 1.3  H (x)_R H = M_4(R)  (exact integer arithmetic)")


def psi_matrix(a, b):
    """4x4 integer matrix of x -> a x conj(b) in the basis 1, i, j, k."""
    cols = [qmul(qmul(a, e), qconj(b)) for e in QBASIS]
    return np.array(cols, dtype=np.int64).T


mats = {(p, q): psi_matrix(QBASIS[p], QBASIS[q]) for p in range(4) for q in range(4)}
flat = np.array([m.ravel() for m in mats.values()], dtype=float)
check("the 16 images psi(e_p (x) e_q) are linearly independent (rank 16)",
      np.linalg.matrix_rank(flat) == 16)
ok = True
for (p1, q1), (p2, q2) in it.product(mats, repeat=2):
    a = qmul(QBASIS[p1], QBASIS[p2]); b = qmul(QBASIS[q1], QBASIS[q2])
    ok &= np.array_equal(psi_matrix(a, b), mats[(p1, q1)] @ mats[(p2, q2)])
check("psi((a1(x)b1)(a2(x)b2)) = psi(a1(x)b1) psi(a2(x)b2) for all 256 basis pairs", ok)
check("psi(1 (x) 1) = I_4", np.array_equal(mats[(0, 0)], np.eye(4, dtype=np.int64)))
check("conjugation reverses products: conj(b1 b2) = conj(b2) conj(b1)",
      all(qconj(qmul(a, b)) == qmul(qconj(b), qconj(a)) for a in QBASIS for b in QBASIS))
check("dim_R(H (x) H) = 16 = dim_R M_4(R)", 4 * 4 == 4 ** 2)

# ------------------------------------------------------------------ Corollary 1.4
section("Corollary 1.4  C_n = H^{(x)n}, Brauer parity and J_n^2")
# [H] != 0: H is a division algebra (norm is multiplicative, so ab = 0 => a = 0 or b = 0)
ok = True
for _ in range(500):
    a = tuple(int(v) for v in rng.integers(-9, 10, 4)); b = tuple(int(v) for v in rng.integers(-9, 10, 4))
    ok &= qnorm2(qmul(a, b)) == qnorm2(a) * qnorm2(b)
check("N(ab) = N(a) N(b) exactly (H has no zero divisors, so [H] != 0)", ok)
note("[H]^2 = [M_4(R)] = 0 (Lemma 1.3) and [H] != 0 (no zero divisors)  =>  [H] has order 2")

for n in range(1, 5):
    prods = [kron(*[phi(QBASIS[i]) for i in idx]) for idx in it.product(range(4), repeat=n)]
    fixed = all(np.allclose(J_apply_conj(n, M), M) for M in prods)
    check(f"n={n}: phi(H)^(x)n is J_n-fixed", fixed)
    check(f"n={n}: phi(H)^(x)n has real dimension 4^n = {4 ** n} (= the whole fixed real form)",
          real_rank(prods) == 4 ** n)

# Type detection: a J_n-fixed rank-1 projection |v><v| needs J_n v = c v.
for n in range(1, 6):
    S = kron(*[SY] * n)
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    v /= np.linalg.norm(v)
    Jv = S @ v.conj()
    if n % 2:
        check(f"n={n} (odd): <v, J_n v> = 0 for random v (Kramers) -> no rank-1 fixed idempotent,"
              f" C_n = M_{2 ** (n - 1)}(H)", abs(np.vdot(v, Jv)) < 1e-12)
    else:
        u = v + Jv
        u /= np.linalg.norm(u)
        P = np.outer(u, u.conj())
        check(f"n={n} (even): u = v + J_n v gives a J_n-fixed rank-1 projection,"
              f" C_n = M_{2 ** n}(R)", np.allclose(J_apply_conj(n, P), P))

raise SystemExit(summary("Section 1"))
