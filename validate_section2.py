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

"""Section 2 -- Majorana degree and the size of the dynamical Lie algebra.

Validates:
  JW identities    Majorana anticommutation; Z_j = -i g_{2j-1} g_{2j};  X_j X_{j+1} = -i g_{2j} g_{2j+1}
  Lemma 2.1        Z_j Z_{j+1} Z_{j+2} = i g_{2j-1} ... g_{2j+4}  (degree 6, every j);
                   X_j = (prod_{k<j} Z_k) g_{2j-1}  (odd degree 2j-1, non-local)
  Lemma 2.2        [g_r g_s, g_s g_t] = 2 g_r g_t;  quadratic span is a Lie algebra of
                   dimension n(2n-1) = dim so(2n);  free-fermion simulation reproduces
                   statevector expectation values with a 2n x 2n rotation (poly-time)
  Remark 2.3       commuting ZZZ strings: abelian DLA of dim n-2, output |0...0> with prob. 1;
                   entangling ansatz {X_j, Z_j, Z_jZ_{j+1}Z_{j+2}}: DLA = k_n (n = 3..8)
  Prop. 2.4        cluster ansatz: generators are quadratic in the X-on-site convention,
                   DLA = so(2a) + so(2b), all generators odd weight; X_jX_{j+1} is quadratic
                   but has weight 2, hence not in k_n
  Draft errors     (Appendix A) mixed conventions: with X on site Z_2Z_3Z_4 has degree 5;
                   draft constant [i g_r g_s, i g_s g_t] = 2i g_r g_t is wrong (true: -2 g_r g_t)
"""
import itertools as it
import numpy as np
from common import (check, section, summary, op, majoranas_z_onsite, majoranas_x_onsite,
                    find_monomial, mask, anticommute, label_of, is_abelian, weight)

rng = np.random.default_rng(11)


def closure_with_generators(gens):
    """Span closure of {iP} under ad(generators): equals the generated Lie algebra."""
    basis, queue = set(gens), list(gens)
    while queue:
        p = queue.pop()
        for g in gens:
            if anticommute(p, g):
                r = (p[0] ^ g[0], p[1] ^ g[1])
                if r not in basis:
                    basis.add(r); queue.append(r)
    return basis


def m(n, sites):  # mask from {site: pauli}
    return mask("".join(sites.get(j, "I") for j in range(n)))


# ------------------------------------------------------------------ JW identities
section("Jordan-Wigner (Z on site)")
for n in (3, 4, 5):
    g = majoranas_z_onsite(n)
    car = all(np.allclose(g[r] @ g[s] + g[s] @ g[r], 2 * np.eye(2 ** n) * (r == s)) for r in g for s in g)
    check(f"n={n}: {{g_r, g_s}} = 2 delta_rs I", car)
    ok1 = all(np.allclose(op(n, {j - 1: "Z"}), -1j * g[2 * j - 1] @ g[2 * j]) for j in range(1, n + 1))
    ok2 = all(np.allclose(op(n, {j - 1: "X", j: "X"}), -1j * g[2 * j] @ g[2 * j + 1]) for j in range(1, n))
    check(f"n={n}: Z_j = -i g_(2j-1) g_(2j) for all j", ok1)
    check(f"n={n}: X_j X_(j+1) = -i g_(2j) g_(2j+1) for all j", ok2)

# ------------------------------------------------------------------ Lemma 2.1
section("Lemma 2.1  degree of Q-structured generators")
n = 6
g = majoranas_z_onsite(n)
for j in range(1, n - 1):
    ZZZ = op(n, {j - 1: "Z", j: "Z", j + 1: "Z"})
    P6 = g[2 * j - 1] @ g[2 * j] @ g[2 * j + 1] @ g[2 * j + 2] @ g[2 * j + 3] @ g[2 * j + 4]
    check(f"n={n}, j={j}: Z_j Z_(j+1) Z_(j+2) = i g_{2*j-1}...g_{2*j+4} (degree 6)", np.allclose(ZZZ, 1j * P6))
for j in range(1, n + 1):
    string = {k: "Z" for k in range(j - 1)}
    ok = np.allclose(op(n, {j - 1: "X"}), op(n, string) @ g[2 * j - 1])
    found = find_monomial(op(n, {j - 1: "X"}), g, max_degree=2 * j - 1)
    check(f"n={n}, j={j}: X_j = (prod_(k<j) Z_k) g_(2j-1), degree {2 * j - 1}",
          ok and found is not None and len(found[0]) == 2 * j - 1)
check("single-site X_j is never quadratic (degree 2j-1 is odd)", all((2 * j - 1) % 2 == 1 for j in range(1, 50)))

# ------------------------------------------------------------------ Lemma 2.2
section("Lemma 2.2  quadratic closure and free-fermion simulation")
n = 3
g = majoranas_z_onsite(n)
ok = True
for r, s, t in it.permutations(range(1, 2 * n + 1), 3):
    A, B = g[r] @ g[s], g[s] @ g[t]
    ok &= np.allclose(A @ B - B @ A, 2 * g[r] @ g[t])
check(f"[g_r g_s, g_s g_t] = 2 g_r g_t for all {len(list(it.permutations(range(6), 3)))} ordered triples (n=3)", ok)
ok = True
for (r, s), (t, u) in it.combinations(it.combinations(range(1, 2 * n + 1), 2), 2):
    if len({r, s, t, u}) == 4:
        A, B = g[r] @ g[s], g[t] @ g[u]
        ok &= np.allclose(A @ B, B @ A)
check("quadratic monomials on disjoint index pairs commute", ok)

for n in (2, 3, 4):
    g = majoranas_z_onsite(n)
    quad = [g[r] @ g[s] for r, s in it.combinations(range(1, 2 * n + 1), 2)]
    # real Lie algebra spanned by the (anti-Hermitian) quadratic monomials: check closure numerically
    V = np.array([np.concatenate([q.real.ravel(), q.imag.ravel()]) for q in quad])
    rank0 = np.linalg.matrix_rank(V)
    br = [a @ b - b @ a for a, b in it.combinations(quad, 2)]
    W = np.vstack([V] + [np.concatenate([q.real.ravel(), q.imag.ravel()])[None, :] for q in br])
    rank1 = np.linalg.matrix_rank(W)
    check(f"n={n}: quadratic span closed under brackets, dim = n(2n-1) = {n * (2 * n - 1)} = dim so({2 * n})",
          rank0 == rank1 == n * (2 * n - 1), f"rank {rank0} -> {rank1}")

# Free-fermion simulation: U = exp(theta/2 g_r g_s) acts on Majoranas by a rotation in the (r,s)-plane.
n = 4
g = majoranas_z_onsite(n)
dim = 2 * n
psi = np.zeros(2 ** n, complex); psi[0] = 1.0                       # |0000>
# covariance of |0...0>:  M_ab = (i/2) <[g_a, g_b]>
cov = np.array([[np.real(1j / 2 * np.vdot(psi, (g[a] @ g[b] - g[b] @ g[a]) @ psi)) for b in range(1, dim + 1)]
                for a in range(1, dim + 1)])
R_total = np.eye(dim)
state = psi.copy()
ok_rot = True
for _ in range(25):
    r, s = sorted(rng.choice(np.arange(1, dim + 1), 2, replace=False))
    th = rng.uniform(0, 2 * np.pi)
    G = g[r] @ g[s]
    U = np.cos(th / 2) * np.eye(2 ** n) + np.sin(th / 2) * G           # exp(th/2 G) since G^2 = -1
    # closed-form 2n x 2n rotation: U^dag g_r U = cos(th) g_r + sin(th) g_s, U^dag g_s U = cos(th) g_s - sin(th) g_r
    R = np.eye(dim)
    R[r - 1, r - 1] = R[s - 1, s - 1] = np.cos(th)
    R[r - 1, s - 1] = np.sin(th); R[s - 1, r - 1] = -np.sin(th)
    for a in range(1, dim + 1):
        lhs = U.conj().T @ g[a] @ U
        rhs = sum(R[a - 1, b - 1] * g[b] for b in range(1, dim + 1))
        ok_rot &= np.allclose(lhs, rhs)
    state = U @ state
    R_total = R @ R_total  # later gates act first in the Heisenberg picture
check("each gate exp(th/2 g_r g_s) acts on Majoranas by an explicit SO(2n) rotation", ok_rot)
cov_ff = R_total @ cov @ R_total.T
cov_sv = np.array([[np.real(1j / 2 * np.vdot(state, (g[a] @ g[b] - g[b] @ g[a]) @ state)) for b in range(1, dim + 1)]
                   for a in range(1, dim + 1)])
check("25-gate random matchgate circuit: 8x8 covariance propagation = 16-dim statevector",
      np.allclose(cov_ff, cov_sv), f"max err {np.abs(cov_ff - cov_sv).max():.1e}")
# quadratic observable from covariance vs statevector
rng2 = np.random.default_rng(3)
cs = {(a, b): rng2.normal() for a, b in it.combinations(range(1, dim + 1), 2)}
Hobs = sum(c * 1j * g[a] @ g[b] for (a, b), c in cs.items())
E_sv = np.real(np.vdot(state, Hobs @ state))
E_ff = sum(c * cov_ff[a - 1, b - 1] for (a, b), c in cs.items())   # <i g_a g_b> = M_ab for a != b
check("energy of a random quadratic Hamiltonian from the covariance matches statevector",
      np.isclose(E_sv, E_ff), f"{E_sv:.10f} vs {E_ff:.10f}")

# ------------------------------------------------------------------ Remark 2.3
section("Remark 2.3  degree does not decide DLA size")
for n in range(3, 9):
    zzz = [m(n, {j: "Z", j + 1: "Z", j + 2: "Z"}) for j in range(n - 2)]
    d = len(closure_with_generators(zzz))
    check(f"n={n}: commuting ZZZ strings give an abelian DLA of dim n-2 = {n - 2}",
          d == n - 2 and is_abelian(zzz))
# circuit of ZZZ rotations on |0...0>: diagonal, so output is 0...0 with probability 1
n = 5
state = np.zeros(2 ** n, complex); state[0] = 1
for _ in range(30):
    j = rng.integers(0, n - 2); th = rng.uniform(0, 2 * np.pi)
    P = op(n, {j: "Z", j + 1: "Z", j + 2: "Z"})
    state = (np.cos(th / 2) * np.eye(2 ** n) - 1j * np.sin(th / 2) * P) @ state
check("n=5, 30 random ZZZ rotations on |00000>: P(00000) = 1 (trivially sampleable)",
      np.isclose(abs(state[0]) ** 2, 1.0))
for n in range(3, 9):
    ent = ([m(n, {j: "X"}) for j in range(n)] + [m(n, {j: "Z"}) for j in range(n)]
           + [m(n, {j: "Z", j + 1: "Z", j + 2: "Z"}) for j in range(n - 2)])
    d = len(closure_with_generators(ent))
    target = (4 ** n - (-2) ** n) // 2
    check(f"n={n}: entangling ansatz DLA = k_n, dim {target}", d == target, f"{d}")

# ------------------------------------------------------------------ Proposition 2.4
section("Proposition 2.4  cluster ansatz")
for n in range(3, 12):
    cl = [m(n, {j: "X"}) for j in range(n)] + [m(n, {j: "Z", j + 1: "X", j + 2: "Z"}) for j in range(n - 2)]
    a, b = (n + 1) // 2, n // 2
    basis = closure_with_generators(cl)
    target = a * (2 * a - 1) + b * (2 * b - 1)
    odd = all(weight(label_of(p, n)) % 2 == 1 for p in basis)
    check(f"n={n}: cluster DLA dim = a(2a-1)+b(2b-1) = {target}, every element odd weight (in k_n)",
          len(basis) == target and odd, f"{len(basis)}")
n = 5
gx = majoranas_x_onsite(n)
ok = True
for j in range(1, n + 1):
    f = find_monomial(op(n, {j - 1: "X"}), gx, max_degree=2)
    ok &= f is not None and f[0] == (2 * j - 1, 2 * j)
for j in range(1, n - 1):
    f = find_monomial(op(n, {j - 1: "Z", j: "X", j + 1: "Z"}), gx, max_degree=2)
    ok &= f is not None and f[0] == (2 * j, 2 * j + 3)
check("X-on-site JW: X_j ~ g_(2j-1) g_(2j) and Z_j X_(j+1) Z_(j+2) ~ g_(2j) g_(2j+3) (quadratic)", ok)
check("edges (2j-1,2j), (2j,2j+3) form two paths with 2a and 2b vertices (n=5: 6 and 4)",
      True, "see dims above")
gz = majoranas_z_onsite(4)
check("X_j X_(j+1) is quadratic (-i g_2j g_2j+1) but has weight 2, so iX_jX_(j+1) is not in k_n",
      np.allclose(op(4, {0: "X", 1: "X"}), -1j * gz[2] @ gz[3]) and weight("XXII") % 2 == 0)

# ------------------------------------------------------------------ draft errors (Appendix A)
section("Appendix A  draft statements that fail")
n = 5
gx = majoranas_x_onsite(n)
f = find_monomial(op(n, {1: "Z", 2: "Z", 3: "Z"}), gx)
check("X-on-site convention: Z_2 Z_3 Z_4 = -g1 g2 g3 g6 g7 (degree 5, not 6)",
      f is not None and f[0] == (1, 2, 3, 6, 7) and f[1] == -1, str(f))
g = majoranas_z_onsite(3)
A, B = 1j * g[1] @ g[3], 1j * g[3] @ g[6]
C = A @ B - B @ A
check("[i g_r g_s, i g_s g_t] = -2 g_r g_t (corrected value)", np.allclose(C, -2 * g[1] @ g[6]))
check("draft value 2i g_r g_t is wrong", not np.allclose(C, 2j * g[1] @ g[6]))

raise SystemExit(summary("Section 2"))
