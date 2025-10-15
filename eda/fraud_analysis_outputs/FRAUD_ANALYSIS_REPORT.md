# Fraud Pattern Analysis Report - AI MedGuard

**Generated:** 2025-10-11 13:08:47

---

## 1. Dataset Overview

- **Total Providers:** 100,000
- **Features Analyzed:** 128
- **Date Range:** Current snapshot

## 2. Fraud Risk Summary

**Risk Score Distribution:**

- Score 0: 86,875 providers (86.88%)
- Score 1: 12,338 providers (12.34%)
- Score 2: 763 providers (0.76%)
- Score 3: 24 providers (0.02%)

**High-Risk Providers (score ≥2):** 787 (0.79%)

## 3. Key Findings

### Fraud Indicators Detected:

1. **Overbilling Patterns:** 7,938 providers with suspiciously low payment ratios
2. **High Volume Providers:** 4,998 providers in top 5% of service volume
3. **Unusual Service Patterns:** 1,000 providers with abnormally high services per beneficiary
4. **Multi-State Operations:** 0 providers operating across multiple states

### Statistical Insights:

- **Median Payment Ratio:** 28.58% (Medicare pays ~29% of charges)
- **Average Services per Provider:** 2236
- **Average Beneficiaries per Provider:** 736
- **Services per Beneficiary:** 3.26

## 4. Visualizations

Generated plots (see `figures/` directory):

1. **feature_distributions.png** - Distribution of key features
2. **outlier_boxplots.png** - Outlier detection via box plots
3. **fraud_risk_analysis.png** - Fraud risk patterns
4. **correlation_heatmap.png** - Feature correlations

## 5. Recommendations for Model Development

### Priority Features for Fraud Detection:

1. **pay_ratio** - Strong indicator of overbilling
2. **svc_per_bene** - Identifies unusual service patterns
3. **total_services** - Volume-based anomalies
4. **distinct_hcpcs** - Procedure diversity indicators

### Modeling Approaches:

- **Supervised:** Use fraud_risk_score as proxy labels
- **Unsupervised:** Anomaly detection (Isolation Forest, LOF)
- **Semi-supervised:** Combine labeled high-risk with clustering

### Data Quality Notes:

- All core features have 0% missing values
- Outliers identified and documented
- Feature distributions show expected healthcare patterns

---

*End of Report*
