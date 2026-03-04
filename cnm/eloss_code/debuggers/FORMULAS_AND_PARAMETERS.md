# ARLEO-PEIGNÉ FORMULA COMPARISON
## Finding the d+Au Broadening Bug

---

## FROM PAPER (Images):

### Paper Parameters (Table, J/ψ at √s=200 GeV):
- p₀ = 3.3 GeV
- n = (8.3)  # parentheses in paper - CHECK IF SHOULD BE DIFFERENT!
- m = 4.3

### Paper L_eff values (Footnote 3):
- L_Au = 10.23 fm (Min-Bias, d+Au)
- L_Pb = 10.11 fm (Min-Bias, p+Pb)  
- L_p = 1.5 fm (coherence length)

### Paper pp cross section (Eq. 2.11):
```latex
\frac{d\sigma_{pp}^\psi}{dy d^2p_\perp} = \mathcal{N} \times \left(\frac{p_0^2}{p_0^2 + p_\perp^2}\right)^m \times \left(1 - \frac{2M_\perp}{\sqrt{s}}\cosh y\right)^n
\equiv \mathcal{N} \times \mu(p_\perp) \times \nu(y, p_\perp)
```

where:
- μ(p⊥) = (p₀²/(p₀² + p⊥²))^m  → This is our F1
- ν(y, p⊥) = (1 - 2M⊥/√s cosh y)^n  → This is our F2  
- M⊥ = √(p⊥² + M_ψ²)

### Paper Broadening Formula (Image 3):
```latex
R_{pA}^{broad}(y, p_\perp) \equiv \int_\varphi \frac{\mu(|\vec{p}_\perp - \Delta\vec{p}_\perp|)}{\mu(p_\perp)} \frac{\nu(E, \vec{p}_\perp - \Delta\vec{p}_\perp)}{\nu(E, p_\perp)}
```

**CRITICAL QUESTION:** What is E here? Is it:
a) E = energy (not rapidity y)?
b) Or is it still y but different notation?

---

## OUR IMPLEMENTATION:

### Current code (R_broad):
```python
# F1 ratio
F1_num = (p0² / (p0² + pshift²))^m
F1_den = (p0² / (p0² + pT²))^m
R1 = F1_num / F1_den

# F2 ratio  
arg_num = 1.0 - (2*mT_shift/√s)*cosh(y)  # SAME y for num and den!
arg_den = 1.0 - (2*mT/√s)*cosh(y)
F2_num = arg_num^n
F2_den = arg_den^n
R2 = F2_num / F2_den

R_broad = ∫ dφ/(2π) * R1 * R2
```

---

## POTENTIAL BUG:

Looking at the paper formula, it says:
```
ν(E, p⃗_⊥ - Δp⃗_⊥) / ν(E, p_⊥)
```

The **E argument is THE SAME** in numerator and denominator!

But in the paper Eq 2.11, ν is defined as:
```
ν(y, p_⊥) = (1 - 2M_⊥/√s cosh y)^n
```

So either:
1. The paper is using **E** (energy) instead of **y** (rapidity) in the broadening formula
2. Or it's just notation and E ≡ y

---

## HYPOTHESIS: The ν function might be evaluated DIFFERENTLY

Let me check the p+Pb implementation to see how THEY do it...

If p+Pb uses the **same y** in both F2_num and F2_den (like we do), 
and it WORKS for p+Pb, then this is NOT the bug.

But if the paper formula means something else, we need to understand it!

---

## ACTION ITEMS:

1. Check p+Pb eloss_cronin.py line-by-line vs our d+Au
2. Verify if there's a difference in how F2 is evaluated  
3. Check if the (8.3) in parentheses means n should be different
4. Verify L_d value for d+Au (we use 6.34 fm estimated)

---

## NEXT: Compare p+Pb implementation EXACTLY
