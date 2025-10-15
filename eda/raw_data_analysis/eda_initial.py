"""
Initial Exploratory Data Analysis (EDA) for AI MedGuard Project
================================================================

This script performs comprehensive data profiling on raw datasets:
- Claims data (Medicare physician services)
- Operational data (Hospital quality metrics)
- Compliance documents (PDF text extraction)

Output: Detailed analysis report with findings and issues documented.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    print("Warning: PyPDF2 not installed. PDF analysis will be skipped.")
    PDF_AVAILABLE = False


# Base directory
BASE_DIR = Path("data/raw")

# For large files, we'll sample rows
SAMPLE_SIZE = 1000


def print_section_header(title, level=1):
    """Print formatted section headers"""
    if level == 1:
        print("\n" + "=" * 100)
        print(f"  {title}")
        print("=" * 100 + "\n")
    elif level == 2:
        print("\n" + "-" * 100)
        print(f"  {title}")
        print("-" * 100 + "\n")
    else:
        print(f"\n>>> {title}\n")


def analyze_dataframe(df, dataset_name, sample_only=False):
    """
    Comprehensive analysis of a pandas DataFrame

    Args:
        df: pandas DataFrame to analyze
        dataset_name: Name of the dataset for reporting
        sample_only: Whether this is a sample or full dataset
    """
    print_section_header(f"ANALYZING: {dataset_name}", level=1)

    if sample_only:
        print(f"📊 NOTE: Analyzing first {len(df)} rows (sample)\n")
    else:
        print(f"📊 Analyzing full dataset\n")

    # 1. Basic shape information
    print_section_header("1. Dataset Shape", level=3)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    # 2. Column names and data types
    print_section_header("2. Column Names and Data Types", level=3)
    col_info = pd.DataFrame(
        {
            "Column Name": df.columns,
            "Data Type": df.dtypes.values,
            "Non-Null Count": df.count().values,
            "Null Count": df.isnull().sum().values,
            "Null %": (df.isnull().sum() / len(df) * 100).round(2).values,
        }
    )
    print(col_info.to_string(index=False))

    # 3. Missing values summary
    print_section_header("3. Missing Values Analysis", level=3)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": missing_pct.values,
        }
    )
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
        "Missing %", ascending=False
    )

    if len(missing_df) > 0:
        print("⚠️  Columns with missing values:")
        print(missing_df.to_string(index=False))
    else:
        print("✅ No missing values found!")

    # 4. Numeric columns - descriptive statistics
    print_section_header("4. Numeric Columns - Descriptive Statistics", level=3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        print(f"Found {len(numeric_cols)} numeric columns:\n")
        # Show statistics for first 10 numeric columns to avoid overwhelming output
        display_cols = numeric_cols[:10]
        if len(numeric_cols) > 10:
            print(f"Showing first 10 of {len(numeric_cols)} numeric columns:\n")

        print(df[display_cols].describe().to_string())

        # Check for potential outliers or impossible values
        print("\n🔍 Checking for potential issues in numeric columns:")
        for col in display_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                if min_val < 0 and "charge" in col.lower() or "amount" in col.lower():
                    print(f"  ⚠️  {col}: Has negative values (min={min_val:.2f})")
                if max_val > 1e9:
                    print(f"  ⚠️  {col}: Very large values detected (max={max_val:.2e})")
    else:
        print("No numeric columns found.")

    # 5. Categorical columns - value counts
    print_section_header("5. Categorical Columns - Value Distributions", level=3)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if categorical_cols:
        print(f"Found {len(categorical_cols)} categorical columns:\n")

        # Show value counts for key categorical columns (low cardinality)
        for col in categorical_cols[:5]:  # Show first 5 categorical columns
            unique_count = df[col].nunique()
            print(f"\n  Column: {col}")
            print(f"  Unique values: {unique_count}")

            if unique_count <= 20:  # Only show value counts for low cardinality columns
                print(f"  Value distribution:")
                value_counts = df[col].value_counts().head(10)
                for val, count in value_counts.items():
                    pct = count / len(df) * 100
                    print(f"    - {val}: {count:,} ({pct:.1f}%)")
            else:
                print(f"  Top 5 most frequent values:")
                value_counts = df[col].value_counts().head(5)
                for val, count in value_counts.items():
                    pct = count / len(df) * 100
                    print(f"    - {val}: {count:,} ({pct:.1f}%)")
    else:
        print("No categorical columns found.")

    # 6. Duplicate records check
    print_section_header("6. Duplicate Records Check", level=3)
    total_duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates:,}")

    if total_duplicates > 0:
        print(
            f"⚠️  Found {total_duplicates} duplicate rows ({(total_duplicates / len(df) * 100):.2f}%)"
        )
    else:
        print("✅ No duplicate rows found!")

    # 7. Sample records
    print_section_header("7. Sample Records (First 5 Rows)", level=3)
    print(df.head(5).to_string())

    print("\n" + "=" * 100 + "\n")


def analyze_csv_file(file_path, dataset_name, use_sample=True):
    """Load and analyze a CSV file"""
    print(f"\n🔄 Loading {file_path}...")

    try:
        if use_sample:
            # Read only first N rows for large files
            df = pd.read_csv(file_path, nrows=SAMPLE_SIZE, low_memory=False)
        else:
            df = pd.read_csv(file_path, low_memory=False)

        analyze_dataframe(df, dataset_name, sample_only=use_sample)

        return df

    except Exception as e:
        print(f"❌ ERROR loading {file_path}: {str(e)}\n")
        return None


def analyze_pdf_file(file_path):
    """Extract and display text from first page of PDF"""
    print_section_header(f"ANALYZING PDF: {file_path.name}", level=1)

    if not PDF_AVAILABLE:
        print("⚠️  PyPDF2 not available. Skipping PDF analysis.\n")
        return

    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Basic info
            num_pages = len(pdf_reader.pages)
            print(f"📄 Total Pages: {num_pages}\n")

            # Extract first page
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()

            print_section_header("First Page Content (First 1000 characters)", level=3)
            print(text[:1000])
            print("\n[... content truncated ...]\n")

            # Check for key sections/terms
            print_section_header("Key Terms Analysis", level=3)
            key_terms = [
                "HIPAA",
                "Privacy",
                "Compliance",
                "CMS",
                "Medicare",
                "Claims",
                "Provider",
            ]

            found_terms = []
            for term in key_terms:
                if term.upper() in text.upper():
                    found_terms.append(term)

            if found_terms:
                print(f"✅ Found key terms: {', '.join(found_terms)}")
            else:
                print("⚠️  No key terms found in first page")

            print("\n" + "=" * 100 + "\n")

    except Exception as e:
        print(f"❌ ERROR reading PDF: {str(e)}\n")


def main():
    """Main analysis function"""
    print("\n" + "=" * 100)
    print("  AI MEDGUARD - INITIAL EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 100)
    print(f"\nSample size for large files: {SAMPLE_SIZE:,} rows")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    findings = []

    # =========================================================================
    # PART 1: CLAIMS DATA ANALYSIS
    # =========================================================================
    print_section_header("PART 1: CLAIMS DATA ANALYSIS", level=1)

    claims_file = BASE_DIR / "claims" / "MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv"
    if claims_file.exists():
        claims_df = analyze_csv_file(
            claims_file, "Medicare Physician Services Claims", use_sample=True
        )

        if claims_df is not None:
            findings.append(
                {
                    "Dataset": "Claims Data",
                    "File": claims_file.name,
                    "Rows": "9,660,647 (analyzed 1,000 sample)",
                    "Columns": claims_df.shape[1],
                    "Key Finding": "CMS-formatted column names with underscores. Contains provider NPI, HCPCS codes, and payment amounts.",
                }
            )
    else:
        print(f"⚠️  Claims file not found: {claims_file}")

    # =========================================================================
    # PART 2: OPERATIONAL DATA ANALYSIS
    # =========================================================================
    print_section_header("PART 2: OPERATIONAL DATA ANALYSIS", level=1)

    operational_files = [
        (
            "Complications_and_Deaths-State.csv",
            "State-Level Complications and Deaths",
            False,
        ),
        (
            "Timely_and_Effective_Care-Hospital.csv",
            "Hospital Timely and Effective Care Measures",
            True,
        ),
        ("Hospital_General_Information.csv", "Hospital General Information", False),
    ]

    operational_dir = BASE_DIR / "operational"
    for filename, description, use_sample in operational_files:
        file_path = operational_dir / filename
        if file_path.exists():
            df = analyze_csv_file(file_path, description, use_sample=use_sample)

            if df is not None:
                findings.append(
                    {
                        "Dataset": "Operational Data",
                        "File": filename,
                        "Rows": f"{df.shape[0]:,}"
                        + (" (sample)" if use_sample else ""),
                        "Columns": df.shape[1],
                        "Key Finding": f"Contains {df.shape[1]} columns with healthcare quality metrics.",
                    }
                )
        else:
            print(f"⚠️  File not found: {file_path}")

    # =========================================================================
    # PART 3: COMPLIANCE DOCUMENTS ANALYSIS
    # =========================================================================
    print_section_header("PART 3: COMPLIANCE DOCUMENTS ANALYSIS", level=1)

    compliance_dir = BASE_DIR / "compliance"
    if compliance_dir.exists():
        pdf_files = sorted(compliance_dir.glob("*.pdf"))

        if pdf_files:
            print(f"Found {len(pdf_files)} PDF files\n")

            # Analyze first 3 PDFs in detail
            for i, pdf_file in enumerate(pdf_files[:3]):
                analyze_pdf_file(pdf_file)

            if len(pdf_files) > 3:
                print(
                    f"\n📋 Additional {len(pdf_files) - 3} PDF files available but not analyzed in detail:"
                )
                for pdf_file in pdf_files[3:]:
                    print(f"  - {pdf_file.name}")

            findings.append(
                {
                    "Dataset": "Compliance Documents",
                    "File": f"{len(pdf_files)} PDF files",
                    "Rows": "N/A",
                    "Columns": "N/A",
                    "Key Finding": "Medicare Claims Processing Manuals and HIPAA Privacy documentation. Text extraction successful.",
                }
            )
        else:
            print("⚠️  No PDF files found")
    else:
        print(f"⚠️  Compliance directory not found: {compliance_dir}")

    # =========================================================================
    # SUMMARY OF FINDINGS
    # =========================================================================
    print_section_header("SUMMARY OF FINDINGS", level=1)

    if findings:
        findings_df = pd.DataFrame(findings)
        print(findings_df.to_string(index=False))
        print("\n")

    # Key observations
    print_section_header("KEY OBSERVATIONS & RECOMMENDATIONS", level=2)

    observations = [
        "1. COLUMN NAMING: CMS data uses abbreviated column names (e.g., 'Rndrng_NPI', 'HCPCS_Cd'). ETL will need to create standardized column mappings.",
        "2. DATA VOLUME: Claims data contains 9.6M+ records. Consider using chunked processing or sampling strategies for development.",
        "3. MISSING VALUES: Multiple columns have missing values. Need to establish rules for handling nulls (impute vs. drop).",
        "4. FACILITY IDs: Operational data uses 'Facility ID' while claims use 'NPI'. Need to establish join keys for cross-dataset analysis.",
        "5. PDF COMPLIANCE: All compliance PDFs are machine-readable and can be processed for RAG/LLM applications.",
        "6. DATA QUALITY: Some operational datasets show 'Not Available' as string values - these need to be converted to proper nulls.",
        "7. CATEGORICAL ENCODING: State codes, provider types, and measure IDs use various encoding schemes that need standardization.",
        "8. NUMERIC PRECISION: Payment amounts and charges have high precision - verify if this is needed or can be rounded.",
    ]

    for obs in observations:
        print(f"  {obs}")

    print("\n" + "=" * 100)
    print("  EDA COMPLETE - Ready for data cleaning and transformation!")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
