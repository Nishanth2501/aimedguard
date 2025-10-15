import pandas as pd
import os
import glob
from pathlib import Path

try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    print("Warning: PyPDF2 not installed. PDF validation will be skipped.")
    PDF_AVAILABLE = False

# --- Base directory ---
BASE_DIR = Path("data/raw")

# --- Expected columns for different dataset types ---
EXPECTED_CLAIMS_COLS = [
    "NPI",
    "Provider Name",
    "HCPCS Code",
    "Average Submitted Charge Amount",
    "Street Address 1",
    "City",
    "State",
    "Zip Code",
    "Provider Type",
    "HCPCS Description",
    "Number of Services",
    "Average Medicare Payment Amount",
]

EXPECTED_OPERATIONAL_COLS = [
    "Provider ID",
    "Hospital Name",
    "Measure Name",
    "Score",
    "State",
]

# Keywords to look for in compliance PDFs
COMPLIANCE_KEYWORDS = ["HIPAA", "Privacy", "Compliance", "CMS"]


def check_csv_file(file_path, expected_columns, dataset_type):
    """
    Validate CSV/Excel files for the AI MedGuard project.

    Args:
        file_path: Path to the CSV file
        expected_columns: List of required column names
        dataset_type: Type of dataset (e.g., "Claims", "Operational")
    """
    print(f"\n{'=' * 80}")
    print(f"Checking {dataset_type} File: {file_path}")
    print(f"{'=' * 80}")

    try:
        # Read the file
        if str(file_path).endswith(".xlsx") or str(file_path).endswith(".xls"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        # Print shape
        print(f"\n📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Print column names
        print(f"\n📋 Column Names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

        # Check for missing required columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            print(f"\n⚠️  Missing Required Columns: {missing_cols}")
        else:
            print(f"\n✅ All required columns present!")

        # Print first 5 rows
        print(f"\n📄 First 5 Data Rows:")
        print(df.head(5).to_string())

        print(f"\n{'=' * 80}\n")
        return True

    except FileNotFoundError:
        print(f"❌ ERROR: File not found - {file_path}")
        print(f"{'=' * 80}\n")
        return False
    except Exception as e:
        print(f"❌ ERROR loading file: {str(e)}")
        print(f"{'=' * 80}\n")
        return False


def check_pdf_file(file_path):
    """
    Validate PDF files for compliance documentation.

    Args:
        file_path: Path to the PDF file
    """
    print(f"\n{'=' * 80}")
    print(f"Checking PDF File: {file_path}")
    print(f"{'=' * 80}")

    if not PDF_AVAILABLE:
        print("⚠️  WARNING: PyPDF2 not available. Skipping PDF validation.")
        print(f"{'=' * 80}\n")
        return False

    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"\n📄 Total Pages: {num_pages}")

            # Extract first page text
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()

            # Print first 500 characters
            print(f"\n📝 First 500 characters of page 1:")
            print("-" * 80)
            print(text[:500])
            print("-" * 80)

            # Check for compliance keywords
            found_keywords = [
                kw for kw in COMPLIANCE_KEYWORDS if kw.upper() in text.upper()
            ]

            if found_keywords:
                print(f"\n✅ Found compliance keywords: {', '.join(found_keywords)}")
            else:
                print(
                    f"\n⚠️  WARNING: None of the expected keywords found: {', '.join(COMPLIANCE_KEYWORDS)}"
                )

            print(f"\n{'=' * 80}\n")
            return True

    except FileNotFoundError:
        print(f"❌ ERROR: File not found - {file_path}")
        print(f"{'=' * 80}\n")
        return False
    except Exception as e:
        print(f"❌ ERROR reading PDF: {str(e)}")
        print(f"{'=' * 80}\n")
        return False


def main():
    """Main function to validate all raw data files."""
    print("\n" + "=" * 80)
    print("AI MEDGUARD - RAW DATA VALIDATION")
    print("=" * 80)

    results = {"claims": [], "operational": [], "compliance": []}

    # Check Claims data
    print("\n\n🏥 VALIDATING CLAIMS DATA")
    print("-" * 80)
    claims_dir = BASE_DIR / "claims"
    if claims_dir.exists():
        for file_path in claims_dir.glob("*.csv"):
            success = check_csv_file(file_path, EXPECTED_CLAIMS_COLS, "Claims")
            results["claims"].append((file_path.name, success))
        for file_path in claims_dir.glob("*.xlsx"):
            success = check_csv_file(file_path, EXPECTED_CLAIMS_COLS, "Claims")
            results["claims"].append((file_path.name, success))
    else:
        print(f"⚠️  Claims directory not found: {claims_dir}")

    # Check Operational data
    print("\n\n🏥 VALIDATING OPERATIONAL DATA")
    print("-" * 80)
    operational_dir = BASE_DIR / "operational"
    if operational_dir.exists():
        for file_path in operational_dir.glob("*.csv"):
            success = check_csv_file(
                file_path, EXPECTED_OPERATIONAL_COLS, "Operational"
            )
            results["operational"].append((file_path.name, success))
        for file_path in operational_dir.glob("*.xlsx"):
            success = check_csv_file(
                file_path, EXPECTED_OPERATIONAL_COLS, "Operational"
            )
            results["operational"].append((file_path.name, success))
    else:
        print(f"⚠️  Operational directory not found: {operational_dir}")

    # Check Compliance PDFs
    print("\n\n📋 VALIDATING COMPLIANCE DOCUMENTS")
    print("-" * 80)
    compliance_dir = BASE_DIR / "compliance"
    if compliance_dir.exists():
        pdf_files = list(compliance_dir.glob("*.pdf"))
        if pdf_files:
            for file_path in pdf_files:
                success = check_pdf_file(file_path)
                results["compliance"].append((file_path.name, success))
        else:
            print("⚠️  No PDF files found in compliance directory")
    else:
        print(f"⚠️  Compliance directory not found: {compliance_dir}")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for category, files in results.items():
        if files:
            print(f"\n{category.upper()}:")
            for filename, success in files:
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"  {status} - {filename}")
        else:
            print(f"\n{category.upper()}: No files checked")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
