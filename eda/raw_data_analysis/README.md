# Raw Data Analysis - AI MedGuard

This folder contains exploratory data analysis (EDA) for the **raw, unprocessed data**.

## Files

### Analysis Scripts

- **`eda_initial.py`** - Comprehensive EDA script for all raw datasets
  - Analyzes claims, operational, and compliance data
  - Generates detailed statistics and data quality reports
  - Identifies issues before ETL processing

- **`rawdata.py`** - Data validation script
  - Validates file existence and structure
  - Checks column names and data types
  - Extracts and validates PDF documents

### 📄 Output Files

- **`eda_report.txt`** - Full EDA analysis output (43KB)
  - Complete statistical summaries
  - Missing value analysis
  - Sample data views

## How to Use

### Run Raw Data EDA
```bash
# From project root
cd /Users/nishanthreddypalugula/Projects/AiMedguard
python eda/raw_data_analysis/eda_initial.py
```

### Run Data Validation
```bash
python eda/raw_data_analysis/rawdata.py
```

## What This Analysis Reveals

### Claims Data (Raw)
- 9,660,647 records with CMS-formatted column names
- Column naming uses underscores (e.g., `Rndrng_NPI`, `HCPCS_Cd`)
- Missing values in expected places (address line 2, middle initials)
- Duplicate credential encoding (M.D. vs MD)

### Operational Data (Raw)
- "Not Available" strings instead of proper nulls
- Mixed data types in Score column
- State-level and hospital-level metrics
- 5,381 unique hospitals across 56 states

### Compliance Data (Raw)
- 9 PDF files (1,559 pages total)
- All PDFs machine-readable
- Comprehensive Medicare and HIPAA documentation

## Key Findings

### Data Quality Issues Identified:
1. "Not Available" string encoding issues
2. Duplicate credential formats
3. Mixed data types in some columns
4. CMS abbreviations in column names
5. No duplicate records found
6. Consistent data structure

## Documentation

For detailed findings, see:
- `/INITIAL_DATA_UNDERSTANDING.md` - Complete data dictionary
- `/DATA_INSIGHTS.md` - Business insights and patterns
- `/EDA_COMPLETION_SUMMARY.md` - Executive summary

## Next Steps

After reviewing raw data issues:
1. Run ETL pipelines to clean data → `make clean_step2`
2. Analyze processed data → see `../processed_data_analysis/`
3. Compare before/after transformations

---

**Analysis Status:** Complete  
**Data Issues:** 6 identified, all documented  
**Ready for:** ETL processing

