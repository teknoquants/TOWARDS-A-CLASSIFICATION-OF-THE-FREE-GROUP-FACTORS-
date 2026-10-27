# Validation scripts — Towards a Classification of the Free Group Factors L_𝕂(F_k) over ℝ, ℂ and ℍ as Real von Neumann Algebras

Roger Cooperman and A.E. Petrunina V.V. · Vienna · 27 October 2026 · MIT License (see `LICENSE`)

Plain Python 3 (3.9 or later) and NumPy. No other dependencies.

```
python run_all.py            # all sections, about 30 seconds, prints a PASS/FAIL summary
python validate_section2.py  # any single section
```

The suite runs 173 checks, and all of them pass. `full_output.txt` is a complete log of one run.

## What each script checks

| Paper result | Script | How it is checked |
|---|---|---|
| Lemma 1.1, J_n² = (−1)ⁿ I | section1 | Matrix computation, n = 1–7 |
| Prop. 1.2, odd-weight criterion | section1 | J P J⁻¹ = (−1)^w P for every Pauli string, n ≤ 4 |
| Prop. 1.2, dim 𝔨_n = (4ⁿ − (−2)ⁿ)/2, sp/so type | section1 | Count = formula = dim sp/so, n ≤ 9; closure under brackets; dimension of the fixed subalgebra by direct linear algebra; invariant bilinear form symmetric (so) or antisymmetric (sp) |
| φ: ℚ → M₂(ℂ), ℍ ⊗ ℂ ≅ M₂(ℂ) | section1, section4 | Homomorphism, image in SU(2), complex rank 4 |
| Lemma 1.3, ℍ ⊗ ℍ ≅ M₄(ℝ) | section1 | **Exact integer arithmetic**: all 256 basis products, rank 16 |
| Cor. 1.4, Brauer order 2 and the sign of J_n² | section1 | Multiplicative norm (no zero divisors); φ(ℍ)^⊗n is the whole fixed algebra (dim 4ⁿ); rank-1 fixed projections for even n, Kramers ⟨v, J_n v⟩ = 0 for odd n |
| Jordan–Wigner identities | section2 | Majorana anticommutation; Z_j and X_jX_{j+1} as quadratics |
| Lemma 2.1, ZZZ has degree 6; X_j has degree 2j−1 | section2 | Exact monomial identification |
| Lemma 2.2, [γ_rγ_s, γ_sγ_t] = 2γ_rγ_t; so(2n) closure | section2 | All triples; closure dimension n(2n−1) |
| Lemma 2.2, poly-time simulation | section2 | 25-gate random matchgate circuit: 8×8 covariance propagation equals the 16-dimensional statevector |
| Remark 2.3 | section2 | Commuting ZZZ: abelian DLA of dimension n−2 and deterministic output; entangling DLA = 𝔨_n for n = 3–8 (up to 32,640 dimensions) |
| Prop. 2.4, cluster ansatz | section2 | DLA = so(2a) ⊕ so(2b) for n = 3–11, all elements of odd weight; generators quadratic in the X-on-site convention; X_jX_{j+1} has weight 2 |
| Thm 3.1, amplification law | section3 | Exact rationals: group action, inverse, Voiculescu integer case, M_d(M) = M_d |
| Thm 3.2, dichotomy | section3 | **Machine-checked proof certificate**: from a hypothetical L(F_x) ≅ L(F_y), builds and independently re-verifies a chain of isomorphisms L(F_a) ≅ … ≅ L(F_b) (300 random cases plus edge cases) |
| Thm 3.2, fundamental group; the draft's s | section3 | t² = (y−1)/(x−1) works; the draft's s sends 3 to 5 instead of 2 |
| Lemma 4.1, (k+3)/4 | section4 | Exact arithmetic, k = 2–1000 and random rationals |
| Prop. 4.2, complexification | section4 | (A ⊗ ℍ) ⊗ ℂ = M₂(A ⊗ ℂ) on the finite-dimensional real group algebra ℝ[S₃]; exact parameter chain to L(F₂) |
| Thm 4.3 (1)–(3) | section4 | ℍ vs M₂(ℝ); M₄(ℝ) vs M₂(ℍ) separated by minimal left-ideal dimension (4 vs 8); trace compatibility of the hyperfinite regrouping |
| Remark 4.4 | section4 | Monte Carlo of GOE, GUE and GSE 4th moments: fitted 1/N coefficients ≈ +5.00, +0.02, −2.48 |
| Section 5, d_β(N) | section5 | Real dimension from explicit bases, N ≤ 5 |
| Section 5, the draft's claims fail | section5 | Lie-algebra dimensions ≠ d_β; log-volume gap of order N² log N, not O(1/N); (S₁, S₂, S₁) counterexample mechanism |
| Section 5, asymptotic freeness | section5 | tr(ABAB) → 0, tr(A²B²) → 1 for β = 1, 2, 4 |
| Appendix A, draft errors | section2, section3, section5 | Each error is reproduced: mixed JW conventions (degree 5), wrong commutator constant, wrong s, O(1/N) estimate |

## What is not, and cannot be, validated by computation

Lines marked `[NOTE]` are logical steps or citations; they are not counted as checks.

- **Theorem 3.1** (Dykema–Rădulescu amplification formula) and **Dykema's free product formula** are theorems about infinite-dimensional von Neumann algebras. The scripts check their algebraic consistency, and they check that Theorem 3.2's proof is correct *given* them. They do not prove the theorems.
- Whether the collapse case of the dichotomy includes r = ∞ is cited, not checked.
- **Problem 2 itself:** whether L(F₂) ≅ L(F₃), and whether L_ℍ(F₅) ≅ L_ℝ(F₂). Both are open; no script claims evidence either way.
- **Conjecture 5.1** (β-invariance of free entropy dimension) is open and is not tested.
- **Remark 4.4** is a numerical observation and is reported as such.

## Files

| File | Contents |
|---|---|
| `common.py` | Pauli and Majorana utilities, exact quaternion arithmetic, Lie closure, check/note reporting |
| `validate_section1.py` … `validate_section5.py` | One script per section of the paper |
| `run_all.py` | Runs everything; exit code 0 only if every check passes |
| `full_output.txt` | Log of a complete run |
| `LICENSE` | MIT License; every script also carries the full licence text in its header |
