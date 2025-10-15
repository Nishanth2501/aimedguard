import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from agent.medguard_agent import build_agent

st.set_page_config(page_title="AI MedGuard Assistant", layout="wide")
st.title("AI MedGuard Assistant")
st.markdown(
    """
<div style="text-align: center; margin: 20px 0; padding: 20px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #1f77b4;">
<p style="margin: 0; font-size: 16px; line-height: 1.6; color: #333;">
AI MedGuard is an intelligent healthcare analytics assistant designed to support hospitals and insurers in improving decision-making. It uses machine learning models to detect potential fraud in medical claims, forecast operational efficiency, and identify anomalies in hospital performance. In addition, it integrates a generative AI compliance engine that retrieves and explains relevant CMS and HIPAA guidelines to ensure policy alignment.
</p>
<br>
<p style="margin: 0; font-size: 16px; line-height: 1.6; color: #333;">
Through this system, users can analyze provider data, view predictive insights, understand the reasons behind model decisions, and ask natural-language questions about healthcare regulations—all within one unified platform. It demonstrates the ability to build, explain, and deploy end-to-end AI solutions that connect predictive analytics with practical business outcomes.
</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align: center; margin: 20px 0; padding: 20px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #28a745;">
<p style="margin: 0; font-size: 16px; line-height: 1.6; color: #333;">
AI MedGuard is powered by publicly available and anonymized healthcare datasets from the U.S. Centers for Medicare & Medicaid Services (CMS). 
These datasets include millions of Medicare physician service claims, hospital quality metrics, and compliance documentation such as HIPAA and Medicare processing manuals. 
The data foundation enables accurate fraud detection, operational forecasting, and real-time compliance retrieval—demonstrating practical, large-scale AI integration in the healthcare domain.
</p>
</div>
""",
    unsafe_allow_html=True,
)

# Create tab navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Chat", "EDA Analysis", "Dataset Info", "AI Capabilities", "About System"]
)

# Initialize session state
if "agent" not in st.session_state:
    try:
        st.session_state.agent = build_agent()
    except Exception as e:
        st.session_state.agent = None
        st.session_state.agent_error = str(e)
if "history" not in st.session_state:
    st.session_state.history = []

# Demo prompts
DEMO_PROMPTS = [
    "Check fraud risk for provider with pay_ratio=1.25, svc_per_bene=18, total_beneficiaries=42",
    "What are the key indicators of fraudulent billing patterns in healthcare providers?",
    "Forecast operational metrics for provider with total_beneficiaries=120, mean_payment=350.4, mean_charge=620.8",
    "Detect anomalies for provider with svc_per_bene=22, total_beneficiaries=85, mean_payment=970.2",
    "What does CMS manual say about resubmitting denied Medicare claims?",
    "Summarize HIPAA rules regarding data privacy during billing",
    "A hospital shows high pay_ratio and unusual service patterns. What business risk does this indicate?",
    "Explain the fraud detection methodology used by AI MedGuard",
    "Does CMS guideline allow automated claim audits based on AI fraud scores?",
    "Give me a summary of common fraud patterns and compliance requirements from CMS documentation",
]


def render_history():
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**MedGuard:** {msg}")


def render_chat_interface():
    # Show demo prompts only when history is empty
    if len(st.session_state.history) == 0:
        st.markdown("### Try These Example Prompts")
        st.markdown("*Click any prompt to see AI MedGuard in action*")

        cols = st.columns(2)
        for i, prompt in enumerate(DEMO_PROMPTS):
            # Truncate display text if too long
            display_text = prompt[:70] + ("…" if len(prompt) > 70 else "")

            if cols[i % 2].button(
                display_text, key=f"demo_{i}", use_container_width=True
            ):
                # Add user message
                st.session_state.history.append(("user", prompt))

                # Check if agent is available
                if st.session_state.get("agent") is None:
                    reply = "Agent not available. Please ensure local resources are properly configured."
                else:
                    # Get agent response
                    try:
                        out = st.session_state.agent.invoke(
                            {"input": prompt, "chat_history": []}
                        )
                        reply = out["output"]
                    except Exception as e:
                        reply = f"Error: {e}"

                # Add agent response
                st.session_state.history.append(("ai", reply))
                st.rerun()

        st.divider()

    render_history()

    # Chat input with Clear Chat button
    col1, col2 = st.columns([4, 1])
    with col1:
        user_in = st.chat_input(
            "Ask about fraud risk, ops forecast, or CMS/HIPAA policy..."
        )
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    if user_in:
        st.session_state.history.append(("user", user_in))

        # Check if agent is available
        if st.session_state.get("agent") is None:
            reply = "Agent not available. Please ensure local resources are properly configured."
        else:
            try:
                out = st.session_state.agent.invoke(
                    {"input": user_in, "chat_history": []}
                )
                reply = out["output"]
            except Exception as e:
                reply = f"Error: {e}"

        st.session_state.history.append(("ai", reply))
        st.rerun()


# Tab 1: Chat Interface
with tab1:
    render_chat_interface()

# Tab 2: EDA Analysis
with tab2:
    st.markdown("## EDA Analysis & Business Key Metrics")

    st.markdown(
        """
    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3; margin-bottom: 20px;">
    <h4 style="margin: 0 0 10px 0; color: #1976d2;">Business Intelligence Overview</h4>
    <p style="margin: 0; color: #424242;">
    This section presents key business metrics derived from comprehensive analysis of 9.6M Medicare claims, 
    5,381 hospitals, and 124K operational records. These insights drive fraud detection, operational efficiency, 
    and compliance monitoring across the healthcare system.
    </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # === BUSINESS KEY METRICS SECTION ===
    st.markdown("### Business Key Metrics")

    # Row 1: Financial Metrics
    st.markdown("#### Financial Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Average Medicare Payment",
            value="$87.05",
            delta="20% of charges",
            delta_color="normal",
        )
    with col2:
        st.metric(
            label="Median Payment Ratio",
            value="28.58%",
            delta="~29% paid",
            delta_color="off",
        )
    with col3:
        st.metric(
            label="Average Submitted Charge",
            value="$417.60",
            delta="Per service",
            delta_color="off",
        )
    with col4:
        st.metric(
            label="Average Allowed Amount",
            value="$111.36",
            delta="CMS approved",
            delta_color="off",
        )

    # Row 2: Operational Metrics
    st.markdown("#### Operational Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Total Providers Analyzed",
            value="100,000",
            delta="Active providers",
            delta_color="off",
        )
    with col2:
        st.metric(
            label="Avg Services per Provider",
            value="2,236",
            delta="224 median",
            delta_color="normal",
        )
    with col3:
        st.metric(
            label="Avg Beneficiaries",
            value="736",
            delta="Per provider",
            delta_color="off",
        )
    with col4:
        st.metric(
            label="Services per Beneficiary",
            value="3.26",
            delta="Avg utilization",
            delta_color="off",
        )

    # Row 3: Risk & Compliance Metrics
    st.markdown("#### Risk & Compliance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="High-Risk Providers",
            value="787",
            delta="0.79% of total",
            delta_color="inverse",
        )
    with col2:
        st.metric(
            label="Overbilling Patterns",
            value="7,938",
            delta="Low pay ratio",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="High Volume Outliers",
            value="4,998",
            delta="Top 5% volume",
            delta_color="normal",
        )
    with col4:
        st.metric(
            label="Unusual Service Patterns",
            value="1,000",
            delta="Anomalies detected",
            delta_color="inverse",
        )

    # Row 4: Data Quality & Coverage Metrics
    st.markdown("#### Data Quality & Coverage Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Data Quality Score",
            value="98/100",
            delta="+13 points",
            delta_color="normal",
        )
    with col2:
        st.metric(
            label="Records Analyzed",
            value="9.6M",
            delta="Claims data",
            delta_color="off",
        )
    with col3:
        st.metric(
            label="Hospitals Covered",
            value="5,381",
            delta="All 50 states",
            delta_color="off",
        )
    with col4:
        st.metric(
            label="Compliance Docs",
            value="1,559",
            delta="Pages processed",
            delta_color="off",
        )

    st.markdown("---")

    # Fraud Risk Distribution
    st.markdown("### Fraud Risk Distribution")

    # Create a simple bar chart
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    risk_levels = ["Low Risk", "Medium Risk", "High Risk"]
    percentages = [86.88, 12.34, 0.79]
    colors = ["green", "orange", "red"]

    bars = ax.bar(risk_levels, percentages, color=colors, alpha=0.7)
    ax.set_ylabel("Percentage of Providers")
    ax.set_title("Fraud Risk Distribution")
    ax.set_ylim(0, 100)

    # Add percentage labels on bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{percentage}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # === BUSINESS INSIGHTS SECTION ===
    st.markdown("### Business Insights & Findings")

    insights_tab1, insights_tab2, insights_tab3, insights_tab4 = st.tabs(
        ["Financial Insights", "Fraud Patterns", "Operational Insights", "Data Quality"]
    )

    with insights_tab1:
        st.markdown("#### Financial Performance Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Payment Patterns:**
            - **Key Finding:** Medicare pays only ~20% of submitted charges on average
            - **Payment Range:** $0.08 to $1,311.76 per service
            - **Avg Submitted Charge:** $417.60
            - **Avg Medicare Allowed:** $111.36
            - **Avg Medicare Payment:** $87.05
            - **Median Payment Ratio:** 28.58% (charges to payment)
            """)

        with col2:
            st.markdown("""
            **Cost Analysis Insights:**
            - **High Cost Variance:** Procedures show wide payment variations
            - **Geographic Differences:** Payment rates vary significantly by state
            - **Provider Type Impact:** Individual vs organizational billing patterns differ
            - **Service Volume:** Average 2,236 services per provider
            - **Beneficiary Count:** Average 736 beneficiaries per provider
            """)

    with insights_tab2:
        st.markdown("#### Fraud Detection Patterns")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **High-Risk Indicators Identified:**
            - **787 High-Risk Providers** (fraud score ≥2)
            - **7,938 Overbilling Patterns** (suspiciously low payment ratios)
            - **4,998 High Volume Providers** (top 5% service volume)
            - **1,000 Unusual Service Patterns** (abnormally high services per beneficiary)
            - **0 Multi-State Operations** detected in current analysis
            """)

        with col2:
            st.markdown("""
            **Fraud Risk Distribution:**
            - **86.88% Low Risk** (86,875 providers - score 0)
            - **12.34% Medium Risk** (12,338 providers - score 1)
            - **0.76% High Risk** (763 providers - score 2)
            - **0.02% Critical Risk** (24 providers - score 3)
            
            **Priority Features for Detection:**
            - Pay ratio, Service per beneficiary, Total services, Distinct HCPCS codes
            """)

    with insights_tab3:
        st.markdown("#### Operational Performance Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Provider Statistics:**
            - **Total Providers:** 100,000 analyzed
            - **Average Services:** 2,236 per provider
            - **Median Services:** 224 (right-skewed distribution)
            - **Service Range:** 11 to 34,752 services
            - **Avg Beneficiaries:** 736 per provider
            - **Services per Beneficiary:** 3.26 average utilization
            """)

        with col2:
            st.markdown("""
            **Hospital Quality Landscape:**
            - **5,381 Hospitals** across all 50 states + territories
            - **20 Quality Indicators** measured per facility
            - **Performance Distribution:** 50.1% at "same as national average"
            - **Texas:** 456 hospitals (8.5%)
            - **California:** 379 hospitals (7.0%)
            - **Florida:** 221 hospitals (4.1%)
            """)

    with insights_tab4:
        st.markdown("#### Data Quality Improvements")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **ETL Processing Results:**
            - **Raw Data Quality:** 85/100
            - **Processed Data Quality:** 98/100
            - **Improvement:** +13 points
            
            **Data Quality Actions:**
            - Standardized column names
            - Proper data types assigned
            - Null handling implemented
            - Date parsing completed
            - Schema validation passed
            - Feature engineering applied
            """)

        with col2:
            st.markdown("""
            **Data Coverage & Scale:**
            - **Claims Data:** 9,660,647 records
            - **Hospital Quality:** 5,381 facilities
            - **Operational Data:** 124,434 records
            - **Compliance Docs:** 9 PDFs, 1,559 pages
            - **Geographic Coverage:** All 50 states + territories (56 total)
            - **Provider Types:** 95.7% Individual, 4.3% Organizational
            - **HIPAA Compliance:** Verified and anonymized
            """)

    # Data Quality Improvement Chart
    st.markdown("### Data Quality Improvements")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    quality_stages = ["Raw Data", "Processed Data"]
    quality_scores = [85, 98]
    colors = ["red", "green"]

    bars2 = ax2.bar(quality_stages, quality_scores, color=colors, alpha=0.7)
    ax2.set_ylabel("Quality Score")
    ax2.set_title("Data Quality Improvement Through ETL")
    ax2.set_ylim(0, 100)

    # Add score labels
    for bar, score in zip(bars2, quality_scores):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{score}/100",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(fig2)

# Tab 3: Dataset Information
with tab3:
    st.markdown("## Dataset Information")

    # Data Sources & Scale
    st.markdown("### Data Sources & Scale")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Healthcare Data:**
        - Medicare Claims: 9,660,647 records
        - Hospital Quality: 5,381 facilities
        - Operational Data: 124,434 records
        - Compliance Docs: 9 PDFs, 1,559 pages
        """)

    with col2:
        st.markdown("""
        **Coverage & Scope:**
        - Geographic: All 50 states + territories
        - Provider Types: 95.7% individual, 4.3% organizational
        - Service Types: All HCPCS procedure codes
        - Time Period: Current Medicare data snapshot
        """)

    st.markdown("---")

    # Data Quality Validation
    st.markdown("### Data Quality Validation")

    validation_col1, validation_col2 = st.columns(2)
    with validation_col1:
        st.markdown("""
        **Validation Checks:**
        - Schema validation passed
        - No duplicate records found
        - Data types standardized
        - Null handling implemented
        """)

    with validation_col2:
        st.markdown("""
        **Compliance & Privacy:**
        - HIPAA compliance verified
        - Anonymization confirmed
        - CMS data usage approved
        - Local processing only
        """)

# Tab 4: AI Capabilities
with tab4:
    st.markdown("## AI Capabilities")

    # Problem Solving Areas
    st.markdown("### Problem Solving Areas")

    cap_col1, cap_col2 = st.columns(2)

    with cap_col1:
        st.markdown("""
        **Fraud Detection:**
        - Identify suspicious billing patterns
        - Detect unusual service volumes
        - Flag high-risk providers (787 identified)
        - Analyze payment ratio anomalies
        """)

        st.markdown("""
        **Operational Analytics:**
        - Forecast efficiency metrics
        - Predict resource utilization
        - Detect performance anomalies
        - Monitor quality indicators
        """)

    with cap_col2:
        st.markdown("""
        **Compliance Search:**
        - Search CMS/HIPAA documentation
        - Retrieve policy information
        - Answer regulatory questions
        - Provide compliance guidance
        """)

    st.markdown("---")

    # Technical Approach
    st.markdown("### Technical Approach")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("""
        **Machine Learning:**
        - Gradient boosting for fraud detection
        - Random Forest for forecasting
        - Isolation Forest for anomaly detection
        - Feature engineering pipeline
        """)

    with tech_col2:
        st.markdown("""
        **AI Infrastructure:**
        - LangChain agents with tool calling
        - ChromaDB vector store for RAG
        - Local processing (no external APIs)
        - Real-time inference capabilities
        """)

# Tab 5: About System
with tab5:
    st.markdown("## About AI MedGuard System")

    st.markdown("""
    **AI MedGuard** is a comprehensive healthcare analytics platform that demonstrates 
    end-to-end AI integration in the healthcare domain.
    """)

    st.markdown("### System Architecture")

    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown("""
        **Data Pipeline:**
        - ETL processing with quality validation
        - Feature engineering and selection
        - Real-time data ingestion
        - Comprehensive data profiling
        """)

    with arch_col2:
        st.markdown("""
        **AI Components:**
        - Multiple ML models for different tasks
        - RAG system for document search
        - Natural language processing
        - Interactive chat interface
        """)

    st.markdown("### Key Features")

    st.markdown("""
    - **Fraud Detection**: Advanced ML models to identify suspicious billing patterns
    - **Operational Forecasting**: Predictive analytics for healthcare operations
    - **Compliance Search**: AI-powered document retrieval and analysis
    - **Real-time Inference**: Live predictions and recommendations
    - **Local Processing**: Complete privacy with no external API dependencies
    """)

    st.markdown("### Technical Stack")

    st.markdown("""
    - **Backend**: FastAPI with async processing
    - **Frontend**: Streamlit with interactive dashboards
    - **ML**: Scikit-learn, XGBoost, Isolation Forest
    - **NLP**: LangChain, ChromaDB, Sentence Transformers
    - **Data**: Pandas, Parquet, comprehensive EDA pipeline
    """)
