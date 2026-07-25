# Spatiotemporal EV Charging Demand Forecasting with Graph Neural Networks & SHAP Explainability

## Overview
MSc dissertation project developing a spatiotemporal forecasting system 
to predict electric vehicle charging demand across 275 urban zones in 
Shenzhen, China. The project combines graph-based deep learning with 
SHAP explainability to produce both accurate forecasts and actionable 
spatial insights for infrastructure planners.

This work addresses a confirmed research gap: no known prior study combines 
spatiotemporal graph-based forecasting with spatial SHAP explainability 
for EV infrastructure planning.

## Key Results
- **Best model:** T-GCN (Temporal Graph Convolutional Network)
- **RMSE:** 0.012294
- **MAPE:** 2.24%
- **Improvement over XGBoost baseline:** 8.8%
- **Surrogate SHAP model R²:** 0.9330
- **Key finding:** POI Lifestyle composition is a stronger predictor 
  of charging demand than raw POI density

## Project Structure
The project spans five analytical Kaggle notebooks:
1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. Baseline Modelling (ARIMA, XGBoost)
4. Deep Learning (GRU, T-GCN)
5. SHAP Explainability

Plus a deployable Streamlit web application (artefact/app.py)

## Models Implemented & Compared
| Model | Type |
|-------|------|
| ARIMA | Statistical baseline |
| XGBoost | Gradient boosting |
| GRU | Recurrent neural network |
| T-GCN | Graph neural network ✓ Best |

## Tools & Libraries
Python · TensorFlow · scikit-learn · SHAP · pandas · NumPy ·
matplotlib · Streamlit · Kaggle (GPU T4 x2)

## Methods
- Spatiotemporal graph construction (275×275 adjacency matrix)
- KD-tree spatial indexing for POI feature engineering
- Graph Convolutional Network + GRU architecture (T-GCN)
- Surrogate XGBoost model for SHAP explainability
- Spatial SHAP analysis identifying dominant demand drivers

## Artefact
A four-tab Streamlit web application providing:
- Interactive demand forecasting by zone
- SHAP feature importance visualisation
- Spatial pattern analysis
- Scenario planning for infrastructure decisions

## Business Application
Results directly support EV charging infrastructure planning by 
identifying which spatial features drive demand — enabling more 
targeted and cost-effective placement of charging stations. The 
SHAP explainability layer makes findings accessible to non-technical 
stakeholders including urban planners and policymakers.

## Why This Matters for the Energy Transition
EV charging demand forecasting is a critical component of smart grid 
management. As EV adoption scales, understanding when and where 
charging demand will occur enables grid operators to balance supply 
and demand more efficiently reducing reliance on fossil fuel 
peaking plants and maximising the value of renewable generation.

## Dataset
UrbanEV Dataset: 275 traffic analysis zones, Shenzhen, China
Period: September 2022 – February 2023

## Academic Context
Module: CEM100 — MSc Dissertation
Supervisor: Dr Mark Snaith
Institution: Robert Gordon University, Aberdeen
Year: 2026

> Note: Dissertation notebooks are currently private pending 
> academic submission (August 2026). The Streamlit application 
> code is available in the artefact/ folder. Full notebooks as well as streamlit artefact
> will be made public following submission.
