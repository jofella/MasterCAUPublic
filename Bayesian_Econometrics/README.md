# Bayesian Estimation of a Dynamic Linear Regression (Unemployment)
*Home Assignment — Econometrics III (WS 2024/25)*

> This repository contains my solution for the Econometrics III home assignment.  
> The goal is to estimate a dynamic linear regression for U.S. unemployment using a
> Normal–Gamma prior and Gibbs sampling, distinguish between calm and recession
> periods, and produce posterior summaries, convergence diagnostics, and a simple
> one-step-ahead forecast.

---

## Overview
We model monthly unemployment (`UNEMP`) as a function of its own lag, distributed lags of industrial production (`INPRO`), CPI inflation (`CPI`), business confidence (`BCONF`), and a COVID-19 dummy:

\[
\text{UNEMP}_t = \mu + \alpha_1\,\text{UNEMP}_{t-1}
+ \sum_{i=1}^{q}\beta_i\,\text{INPRO}_{t-i}
+ \sum_{i=1}^{q}\gamma_i\,\text{CPI}_{t-i}
+ \sum_{i=1}^{q}\phi_i\,\text{BCONF}_{t-i}
+ \lambda\,\text{COVID}_t + e_t,\quad e_t\sim \mathcal{N}(0, H^{-1})
\]

with regime-dependent precision \(H=\mathrm{diag}(h_t)\), where \(h_t=h_1\) in calm periods and \(h_t=h_2\) in recessions.  
Priors: independent Normal–Gamma
\[
\beta \sim \mathcal{N}(\beta_0, V_0),\quad
h_1 \sim \mathcal{G}(1/s_1^2,\nu_1),\quad
h_2 \sim \mathcal{G}(1/s_2^2,\nu_2).
\]

The script loads pre-processed, stationary monthly data and U.S. recession dates, runs a Gibbs sampler, prints posterior summaries, checks MCMC convergence (trace plots), and produces predictive densities under \(h_1\) (calm) and \(h_2\) (recession).

---

## Files
- `HA_EconIII_WS2024_25.m` — Main MATLAB script (data prep, priors, Gibbs call, diagnostics, forecast).
- `indnormgam_posterior.m` — Function implementing the Gibbs sampler for the Normal–Gamma setup.
- `US_macro.mat` — Input data (`dates`, `data`, `recessions`, `VARnames`).
- *(optional)* `HA_EconIII_WS2024_25.pdf` — Written report with answers/discussion.

---

## Requirements
- **MATLAB** R2020b or newer recommended.
- No special toolboxes required (uses base plotting and simple helpers like `recessionplot` if available).
- About 10k–1,000k draws supported; default burn-in and draws are set in the script.

---

## Usage
1. Place all files in the same folder (e.g. `Bayesian_Econometrics/HA_EconIII_WS2024_25/`).
2. Open MATLAB in that folder and run:
   ```matlab
   HA_EconIII_WS2024_25
