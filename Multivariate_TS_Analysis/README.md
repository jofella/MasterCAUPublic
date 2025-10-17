# Multivariate Time Series Analysis (SoSe 2025)
*Home Assignment — Multivariate Time Series Analysis*

> This repository contains my solution for the **Multivariate Time Series Analysis** home assignment (SoSe 2025).  
> The project applies classical **VAR (Vector Autoregression)** techniques to German macroeconomic data, covering model selection, Granger causality, impulse response analysis, and forecasting evaluation.

---

## Overview

The goal of this assignment is to analyze interdependencies among key macroeconomic indicators for Germany and assess the propagation of energy price shocks through the economy.  
The analysis includes:

1. Data transformation and stationarity assessment  
2. Model order selection using AIC and BIC  
3. Granger causality testing  
4. Impulse Response Function (IRF) estimation and interpretation  
5. Out-of-sample forecasting and model evaluation

---

## Model Specification

Monthly German macroeconomic variables included:

- **xHICP** – core inflation (HICP excl. food & energy)  
- **PROD** – industrial production  
- **ORDER** – orders received by industry  
- **BUSEXP** – IFO business expectations  
- **ENERGY** – energy prices  
- **ECBR** – ECB main refinancing rate  

The transformed data are expressed in monthly percentage changes (log-differences), ensuring **stationarity** for VAR estimation.

The VAR(p) model is estimated as:

