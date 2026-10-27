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

"""Section 3 -- interpolated free group factors.

What can and cannot be checked by computer
------------------------------------------
Theorem 3.1 (the Dykema-Radulescu amplification formula) and Dykema's free product
formula L(F_r) * L(F_s) = L(F_{r+s}) are cited theorems about infinite-dimensional
von Neumann algebras; no finite computation proves them. This script therefore:

  1. checks, in exact rational arithmetic, the algebraic consistency of the
     amplification law f_t(r) = 1 + (r-1)/t^2 (it is a group action of R_+^*,
     agrees with Voiculescu's integer compression formula, M_d(M) = M_d), and
  2. machine-checks the LOGIC of the proof of Theorem 3.2: taking the two cited
     formulas as axioms, it builds an explicit certificate (a chain of isomorphisms,
     each justified by one axiom) that L(F_a) = L(F_b) for arbitrary 1 < a < b,
     starting from a single hypothetical isomorphism L(F_x) = L(F_y), and then
     re-verifies every link of that chain independently;
  3. checks the fundamental-group statements and the draft's wrong choice of s.

Exact arithmetic: amplification parameters are handled through T = t^2 in Q.
"""
from fractions import Fraction as Fr
import math
import random
from common import check, note, section, summary

random.seed(5)
INF = "inf"


def amp(r, T):
    """parameter of L(F_r)_t with t^2 = T:  1 + (r-1)/T   (r may be INF)."""
    if r == INF:
        return INF
    return 1 + (r - 1) / T


# ------------------------------------------------------------------ Theorem 3.1 consistency
section("Theorem 3.1  algebraic consistency of r -> 1 + (r-1)/t^2")
ok_action = ok_id = ok_inv = ok_range = True
for _ in range(2000):
    r = Fr(random.randint(101, 2000), 100)
    S, T = Fr(random.randint(1, 900), random.randint(1, 30)), Fr(random.randint(1, 900), random.randint(1, 30))
    ok_action &= amp(amp(r, S), T) == amp(r, S * T)      # (M_s)_t = M_{st}
    ok_id &= amp(r, 1) == r
    ok_inv &= amp(amp(r, T), 1 / T) == r
    ok_range &= amp(r, T) > 1
check("(L(F_r)_s)_t = L(F_r)_(st): f_t o f_s = f_(st)  (2000 random rational cases)", ok_action)
check("f_1 = identity", ok_id)
check("f_(1/t) inverts f_t", ok_inv)
check("r > 1  =>  1 + (r-1)/t^2 > 1  (parameter stays admissible)", ok_range)
ok = all(amp(Fr(n), Fr(1, k * k)) == 1 + k * k * (n - 1) for n in range(2, 30) for k in range(1, 30))
check("t = 1/k recovers Voiculescu: L(F_n)_(1/k) = L(F_(1+k^2(n-1))), integer for integer n, k", ok)
check("t = d integer: M_d(L(F_r)) = L(F_r)_d = L(F_(1+(r-1)/d^2))",
      all(amp(Fr(r), d * d) == 1 + Fr(r - 1, d * d) for r in range(2, 20) for d in range(1, 10)))
check("r = infinity is a fixed point: L(F_inf)_t = L(F_inf) for all t", amp(INF, Fr(7, 3)) == INF)

# ------------------------------------------------------------------ Theorem 3.2 proof certificate
section("Theorem 3.2  machine-checked proof logic (finite parameters)")


def build_chain(x, y, a, b):
    """Return a list of links (p, q, rule, data) proving L(F_a) = L(F_b) from L(F_x) = L(F_y).

    Rules (axioms):
      'free'   : from L(F_x)=L(F_y) and u > 1 derive L(F_{x+u}) = L(F_{y+u})   [free product formula]
      'amp'    : from L(F_p)=L(F_q) and T>0 derive L(F_{f_T(p)}) = L(F_{f_T(q)})  [Theorem 3.1]
      'trans'  : transitivity
    Step 1+2 of the paper combined: to translate w -> w + eps we pick T = d/eps and
    the base point z = 1 + (w-1)*T, then u = z - x must exceed 1.
    """
    d = y - x
    eps_max = d * (a - 1) / x                         # need a > 1 + (x/d) eps
    N = int((b - a) / eps_max) + 2
    eps = (b - a) / N
    links, w = [], a
    for _ in range(N):
        T = d / eps
        z = 1 + (w - 1) * T                           # pre-image of w under f_T
        u = z - x
        links.append((w, w + eps, "free+amp", dict(z=z, u=u, T=T)))
        w = w + eps
    return links


def verify_chain(x, y, a, b, links):
    """Independent re-verification of every link."""
    if links[0][0] != a or links[-1][1] != b:
        return False, "chain does not run from a to b"
    for (p, q, rule, dat), nxt in zip(links, links[1:] + [None]):
        z, u, T = dat["z"], dat["u"], dat["T"]
        if not (u > 1):                                   # free product formula needs u > 1
            return False, f"u={u} <= 1"
        if not (x + u == z and y + u == z + (y - x)):     # L(F_z) = L(F_{z+d}) from L(F_x)=L(F_y)
            return False, "free-product step arithmetic"
        if not (T > 0 and amp(z, T) == p and amp(z + (y - x), T) == q):
            return False, "amplification step arithmetic"
        if nxt is not None and nxt[0] != q:
            return False, "chain broken"
    return True, ""


ok_all, worst = True, ""
cases = 0
for _ in range(300):
    x = Fr(random.randint(101, 900), 100)
    y = x + Fr(random.randint(1, 900), 100)
    a = Fr(random.randint(101, 900), 100)
    b = a + Fr(random.randint(1, 3000), 100)
    links = build_chain(x, y, a, b)
    ok, why = verify_chain(x, y, a, b, links)
    ok_all &= ok
    cases += 1
    if not ok:
        worst = f"x={x}, y={y}, a={a}, b={b}: {why}"
check(f"{cases} random hypotheses L(F_x)=L(F_y) each yield a verified chain L(F_a)=...=L(F_b)",
      ok_all, worst)
# the specific case used in the statement: one isomorphism forces L(F_2) = L(F_3) and conversely
links = build_chain(Fr(5, 4), Fr(7, 4), Fr(2), Fr(3))
ok, why = verify_chain(Fr(5, 4), Fr(7, 4), Fr(2), Fr(3), links)
check(f"example: L(F_5/4) = L(F_7/4) => L(F_2) = L(F_3) via {len(links)} certified links", ok, why)
# hard case: target below the hypothesis (a < x), tiny gap d
links = build_chain(Fr(30), Fr(30) + Fr(1, 10), Fr(3, 2), Fr(5))
ok, why = verify_chain(Fr(30), Fr(30) + Fr(1, 10), Fr(3, 2), Fr(5), links)
check(f"hard case x=30, d=1/10, target a=3/2 < x, b=5: verified with {len(links)} links", ok, why)

# infinity reduction: L(F_x) = L(F_inf) => L(F_{x+u}) = L(F_inf * F_u) = L(F_inf) = L(F_x)
note("L(F_x)=L(F_inf) gives L(F_(x+u)) = L(F_inf)*L(F_u) = L(F_inf) = L(F_x) for u > 1, "
     "which reduces to the finite case checked above")
note("Theorem 3.1, the free product formula, and the inclusion of r = inf in alternative 1 are "
     "cited theorems [19, 20]; this script verifies the proof logic given them")

# ------------------------------------------------------------------ fundamental group
section("Theorem 3.2  fundamental group and the draft's choice of s")
ok = True
for _ in range(1000):
    x = Fr(random.randint(101, 900), 100); y = x + Fr(random.randint(1, 900), 100)
    T = (y - 1) / (x - 1)                               # t^2 = (y-1)/(x-1) > 1
    ok &= amp(y, T) == x and T != 1
check("L(F_x)=L(F_y), x<y  =>  t = sqrt((y-1)/(x-1)) != 1 lies in F(L(F_y))  [f_t(y) = x]", ok)
# rigid case: t in F(L(F_r)) means f_t(r) = r, which forces t = 1 for finite r
ok = all(amp(Fr(r), T) != Fr(r) for r in range(2, 40) for T in (Fr(1, 4), Fr(9, 4), Fr(2), Fr(4, 9)))
check("finite r: f_t(r) = r only for t = 1 (so F = {1} in the rigid alternative)", ok)
# draft's wrong choice: s^2 = (t1-1)/(t2-1)
t1, t2 = Fr(2), Fr(3)
s2_draft = (t1 - 1) / (t2 - 1)
s2_fix = (t2 - 1) / (t1 - 1)
check("draft s = sqrt((t1-1)/(t2-1)) does NOT send t2 to t1",
      amp(t2, s2_draft) != t1, f"gives {amp(t2, s2_draft)} instead of {t1}")
check("corrected s = sqrt((t2-1)/(t1-1)) sends t2 to t1", amp(t2, s2_fix) == t1)

raise SystemExit(summary("Section 3"))
