# Bayesian Estimation of a Dynamic Linear Regression (Unemployment)
*Home Assignment — Econometrics III (WS 2024/25)*

> This repository contains my solution for the Econometrics III home assignment.  
> The goal is to estimate a dynamic linear regression for U.S. unemployment using a Normal–Gamma prior and Gibbs sampling, distinguishing between calm and recession periods, and producing posterior summaries, convergence diagnostics, and a one-step-ahead forecast.

---

## Overview

This project estimates a Bayesian dynamic linear regression model for the U.S. unemployment rate.  
Monthly unemployment (**UNEMP**) is modeled as a function of its own lag, distributed lags of industrial production (**INPRO**), CPI inflation (**CPI**), business confidence (**BCONF**), and a COVID-19 dummy variable.

The model:






The precision term **H** depends on the economic regime:

- **hₜ = h₁** during calm periods  
- **hₜ = h₂** during recessions  

Independent Normal–Gamma priors are assumed:


The precision term **H** depends on the economic regime:

- **hₜ = h₁** during calm periods  
- **hₜ = h₂** during recessions  

Independent Normal–Gamma priors are assumed:

2. Open MATLAB in that folder.  
3. Run the script:
```matlab
HA_EconIII_WS2024_25
