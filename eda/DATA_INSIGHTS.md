# Data Insights & Initial Observations

**Generated:** October 11, 2025  
**Purpose:** Document key patterns, anomalies, and interesting findings from initial EDA

---

## 1. Claims Data Insights

### Provider Distribution Patterns

**Credential Encoding Issue:**
- Discovered duplicate encodings for same credentials:
  - "M.D." (47.9%) vs "MD" (24.0%) = ~72% are physicians
  - "D.O." (2.7%) vs "DO" (3.0%) = ~6% are osteopathic physicians
  - This indicates data entry inconsistency requiring standardization

**Provider Entity Types:**
- Individual Providers: 95.7% (code "I")
- Organizational Providers: 4.3% (code "O")
- This explains why 4.3% of records lack first names

**Geographic Coverage:**
- Data includes providers from all 50 states + territories
- RUCA codes present for rural/urban classification
- 100% of sample providers participated in Medicare (Rndrng_Prvdr_Mdcr_Prtcptg_Ind = 'Y')

### Service and Payment Patterns (from 1,000 record sample)

**Service Volume Statistics:**
- Average services per provider: 224 services
- Median: 42 services (indicates right-skewed distribution)
- Range: 11 to 34,752 services
- High variability suggests mix of specialties and practice sizes

**Payment Amount Analysis:**
- Average submitted charge: $417.60
- Average Medicare allowed: $111.36
- Average Medicare payment: $87.05
- **Key Finding:** Medicare pays only ~20% of submitted charges on average
- Payment rates range from $0.08 to $1,311.76

**Beneficiary Count:**
- Average: 70 beneficiaries per provider-procedure combination
- Median: 31 beneficiaries
- Max: 3,106 beneficiaries (likely high-volume specialists)

### HCPCS Code Patterns

Sample includes various procedure types:
- 99221-99233: Hospital care (initial and subsequent)
- These are E&M (Evaluation and Management) codes
- High frequency suggests data may be enriched with hospital-based providers

---

## 2. Operational Data Insights

### Hospital Quality Landscape

**State-Level Distribution:**
- 56 states/territories represented
- Each state measured across 20 quality indicators
- Alaska (AK) shows consistent pattern: many hospitals in "same" category, few outliers

**Quality Measure Categories:**

1. **Mortality Measures:**
   - MORT_30_AMI: Heart attack mortality
   - MORT_30_CABG: CABG surgery mortality
   - MORT_30_COPD: COPD mortality
   - Hybrid_HWM: Hospital-wide mortality

2. **Complication Measures:**
   - COMP_HIP_KNEE: Hip/knee replacement complications
   - PSI_08-15: Patient Safety Indicators

**Performance Distribution (State-Level):**
- 50.1% of measurements show "0" hospitals performing worse
- 20.0% show "1" hospital performing worse
- Pattern suggests most hospitals cluster around "same as national average"
- Few hospitals significantly outperform or underperform

### Hospital Characteristics

**Hospital Type Distribution:**
- Acute Care Hospitals: Majority
- Emergency services: Most hospitals offer emergency care
- Birthing services: 41.4% offer (58.6% don't)

**Geographic Concentration:**
- Texas: 456 hospitals (8.5%)
- California: 379 hospitals (7.0%)
- Florida: 221 hospitals (4.1%)
- Pattern matches population distribution

**Quality Rating Patterns:**
- Hospital overall ratings: 1-5 stars
- Many hospitals show "Not Available" for specific measures
- Suggests not all measures apply to all hospital types

### Timely & Effective Care Insights

**Measure Types:**
- Emergency Department metrics
- Electronic Clinical Quality Measures (eCQMs)
- Healthcare Personnel Vaccination rates
- Hospital Harm measures

**Alabama Sample Pattern (from 1,000 records):**
- Records heavily weighted toward Alabama hospitals
- Suggests data may be organized geographically
- Birmingham, Dothan, Montgomery are major hospital cities

**Score Encoding Challenge:**
- Score field contains:
  - Numeric values: 0.3 (30%), percentages
  - Categorical: "Not Available", "high", "low"
  - Requires separate parsing logic for different measure types

---

## 3. Data Quality Observations

### Missing Value Patterns

**Expected Missing Values:**
1. Address Line 2 (69.2% missing)
   - Normal: Not all addresses need second line
   - No action needed

2. Middle Initial (34.6% missing)
   - Normal: Many people don't use middle initials
   - No action needed

3. Footnotes (43-93% missing)
   - Normal: Only used for special cases/exceptions
   - No action needed

**Unexpected Patterns:**
1. "Not Available" as string value
   - Should be NULL/NaN
   - Found in: Hospital counts, scores, samples
   - **Impact:** Prevents numeric analysis
   - **Priority:** High - needs immediate fixing

2. First Name missing (4.3%)
   - Correlates exactly with Organizational entities
   - Expected behavior, properly encoded

### Data Type Issues

**Mixed Type Columns Identified:**

1. **Score Column:** (Timely & Effective Care)
   - Contains: 0.3, "high", "low", "Not Available"
   - Reason: Different measure types use different scales
   - Solution: Create separate numeric_score and categorical_score columns

2. **Hospital Count Columns:** (State-Level)
   - Should be integers
   - Contains "Not Available" strings
   - Solution: Convert to nullable integer type (Int64)

3. **Sample Column:**
   - Should be integers
   - Contains "Not Available" strings
   - Solution: Convert to nullable integer type

### Encoding Inconsistencies

**Priority Standardizations:**

1. **Provider Credentials:**
   - M.D. → MD
   - D.O. → DO
   - M.D → MD (missing period)
   - Implement lookup table with fuzzy matching

2. **State Codes:**
   - Already standardized (2-letter abbreviations)
   - No issues found ✓

3. **Date Formats:**
   - Format: MM/DD/YYYY
   - Consistent across datasets ✓

---

## 4. Cross-Dataset Integration Challenges

### Key Linkage Issue: Provider NPI vs Facility ID

**The Challenge:**
- Claims data: Provider-level (individual physicians)
- Operational data: Facility-level (hospitals)
- No direct linkage between NPI and Facility ID

**Potential Solutions:**

1. **Geographic Matching:**
   - Match providers to nearby facilities using ZIP codes
   - Limitation: Providers may work at multiple facilities

2. **External Crosswalk:**
   - CMS provides NPPES (National Provider database)
   - Can link NPIs to facility affiliations
   - Recommended approach for production

3. **Partial Analysis:**
   - Analyze claims and operational data separately
   - Join only where geographic overlap is clear
   - Valid approach for initial MVP

### Date Range Alignment

**Claims Data:**
- File name suggests 2023 data (D23)
- No date columns in structure shown
- Need to verify reporting period

**Operational Data:**
- Includes Start Date and End Date columns
- Date ranges: 2021-2024
- Various measurement periods:
  - Mortality: 07/01/2021 - 06/30/2024 (3 years)
  - Complications: 04/01/2021 - 03/31/2024 (3 years)
  - Vaccination: 07/01/2024 - 09/30/2024 (quarterly)

**Alignment Strategy:**
- Use most recent complete year for analysis
- Clearly document date ranges for each dataset
- Handle partial year data appropriately

---

## 5. Anomaly Detection Opportunities

### Potential Fraud Indicators (Claims)

Based on data structure, possible fraud detection features:

1. **Charge Amount Outliers:**
   - Avg_Sbmtd_Chrg ranges $0.50 - $18,000
   - Unusually high charges for routine procedures
   - Statistical outlier detection applicable

2. **Service Volume Anomalies:**
   - Tot_Srvcs max of 34,752 in sample
   - Impossibly high service counts per day
   - Time-based validation needed

3. **Geographic Anomalies:**
   - Unusual procedure patterns by region
   - Out-of-area billing patterns
   - Requires geographic feature engineering

4. **Payment Discrepancies:**
   - Large gaps between submitted and allowed amounts
   - Systematic overbilling patterns
   - Ratio analysis opportunities

### Quality of Care Red Flags (Operational)

1. **Hospitals "Worse" than Average:**
   - Already flagged in data
   - Correlate with other metrics

2. **Missing Quality Data:**
   - Hospitals with many "Not Available" scores
   - May indicate reporting compliance issues

3. **Vaccination Rates:**
   - HCP_COVID_19 scores as low as 0.3 (30%)
   - Potential compliance monitoring

---

## 6. Feature Engineering Opportunities

### Claims Data Features

**Provider-Level:**
- Total charges per provider per year
- Average payment rate (payment/submitted)
- Service volume trends
- Specialty inference from HCPCS patterns
- Geographic features (urban/rural RUCA)

**Procedure-Level:**
- Procedure frequency by region
- Payment variations by geography
- Beneficiary demographics (if available in full data)

**Temporal (if dates available):**
- Seasonal patterns
- Trending providers
- Payment timing analysis

### Operational Data Features

**Hospital-Level:**
- Composite quality score across measures
- Improvement trends over time
- Peer comparison metrics
- Geographic quality patterns

**Measure-Level:**
- Correlation between different quality measures
- Mortality vs complication rates
- Safety score combinations

**Cross-Dataset:**
- Provider quality scores (if linkage possible)
- Geographic healthcare deserts
- Cost vs quality analysis

---

## 7. Business Questions to Answer

### For Claims Analysis

1. **Cost Analysis:**
   - Which procedures have highest cost variance?
   - Which states have highest/lowest payment rates?
   - What's the payment rate by provider type?

2. **Fraud Detection:**
   - Which providers show unusual billing patterns?
   - Are there geographic fraud hotspots?
   - What HCPCS codes show highest fraud risk?

3. **Provider Patterns:**
   - Distribution of services by specialty
   - Urban vs rural provider differences
   - Medicare participation patterns

### For Operational Analysis

1. **Quality Assessment:**
   - Which states have highest quality hospitals?
   - Correlation between hospital size and quality?
   - Best performing hospital types?

2. **Comparative Analysis:**
   - Hospital rankings by state
   - Quality trends over time
   - Improvement vs decline patterns

3. **Risk Stratification:**
   - Which hospitals need intervention?
   - Predictive modeling for quality decline
   - Resource allocation optimization

### For Compliance

1. **Policy Questions:**
   - Which HCPCS codes require prior authorization?
   - What are HIPAA requirements for data sharing?
   - How to handle claims appeals?

2. **RAG Use Cases:**
   - Automated compliance checking
   - Policy Q&A chatbot
   - Billing guidance system

---

## 8. Visualization Recommendations

### Priority Dashboards

**Claims Dashboard:**
- Geographic heatmap of charges
- Top procedures by volume
- Payment rate distributions
- Provider type breakdown

**Quality Dashboard:**
- State-by-state quality comparison
- Hospital performance trends
- Measure-specific scorecards
- Improvement/decline tracking

**Fraud Detection Dashboard:**
- Anomaly alerts
- High-risk provider list
- Pattern detection visualizations
- Investigation queue

---

## 9. Machine Learning Model Opportunities

### Supervised Learning

1. **Fraud Classification:**
   - Binary: Fraud vs legitimate
   - Features: Charge patterns, volume, geographic
   - Model: XGBoost, LightGBM

2. **Quality Prediction:**
   - Predict hospital ratings
   - Forecast quality decline
   - Features: Historical performance, hospital characteristics

### Unsupervised Learning

1. **Provider Clustering:**
   - Group similar billing patterns
   - Identify specialty groups
   - Anomaly detection

2. **Hospital Segmentation:**
   - Group by performance profiles
   - Identify peer groups
   - Benchmark development

### NLP/LLM

1. **Compliance RAG:**
   - Question answering on policies
   - Automated documentation
   - Policy summarization

2. **HCPCS Description Analysis:**
   - Procedure similarity
   - Code recommendation
   - Billing guidance

---

## 10. Technical Considerations

### Performance Optimization

**For 9.6M Claims Records:**
- Chunked processing (100K rows per chunk)
- Parallel processing where possible
- Consider DuckDB for SQL-style queries
- Sample for iterative development (10%)

**For 120K Operational Records:**
- Can be processed in memory
- Standard pandas operations sufficient
- Group-by operations will be fast

**For PDF Processing:**
- Pre-process once, store in vector DB
- ChromaDB for similarity search
- Batch processing for initial ingestion

### Data Storage Strategy

**Raw Data:**
- Keep original files immutable
- Version control with DVC
- Cloud backup recommended

**Processed Data:**
- Parquet format for efficiency
- Partitioned by date/state for performance
- Compression enabled

**Compliance Data:**
- Vector embeddings in ChromaDB
- Original PDFs preserved
- Metadata indexed

---

## Summary

The initial EDA reveals:

**High-quality source data** with clear structure  
**Manageable data quality issues** requiring systematic cleaning  
**Rich feature engineering opportunities** for ML models  
**Clear fraud detection and quality improvement use cases**  
**Excellent compliance documentation** ready for RAG implementation  

**Next Step:** Implement ETL pipelines to address identified issues and create clean, analysis-ready datasets.

---

*This document should be updated as new insights are discovered during deeper analysis.*

