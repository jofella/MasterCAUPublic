
---

## Option B — Keep LaTeX, but make GitHub render it
Use `$$ ... $$` (display) and `$ ... $` (inline). **Do not indent** those lines and avoid block quotes (`>`).

```markdown
# Bayesian Estimation of a Dynamic Linear Regression (Unemployment)
*Home Assignment — Econometrics III (WS 2024/25)*

This is my solution for the Econometrics III home assignment.

## Overview
The model:
$$
\text{UNEMP}_t=\mu+\alpha_1\text{UNEMP}_{t-1}
+\sum_{i=1}^{q}\beta_i\,\text{INPRO}_{t-i}
+\sum_{i=1}^{q}\gamma_i\,\text{CPI}_{t-i}
+\sum_{i=1}^{q}\phi_i\,\text{BCONF}_{t-i}
+\lambda\,\text{COVID}_t+e_t,\quad e_t\sim\mathcal{N}(0,H^{-1}).
$$

Regime precision:
$h_t=h_1$ (calm), $h_t=h_2$ (recession).

Priors:
$$
\beta\sim\mathcal{N}(\beta_0,V_0),\qquad
h_1\sim\mathcal{G}(1/s_1^2,\nu_1),\qquad
h_2\sim\mathcal{G}(1/s_2^2,\nu_2).
$$

*(Then reuse the same Files / Requirements / Usage / Output sections from Option A.)*
