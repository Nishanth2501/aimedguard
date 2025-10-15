from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from agent.tools.fraud_tool import fraud_predict
from agent.tools.ops_tool import ops_forecast, ops_anomaly
from agent.tools.compliance_tool import compliance_search


class LocalAgent:
    """
    Local agent that uses local models and RAG database.
    Falls back to OpenAI for general questions if API key is available.
    """

    def __init__(self):
        # Only include tools for models that exist
        self.tools = [fraud_predict, ops_anomaly, compliance_search]
        if Path("models/ops_forecast.joblib").exists():
            self.tools.append(ops_forecast)

    def invoke(self, query: str) -> dict:
        """
        Process query using local tools and models.
        """
        query_lower = query.lower()

        # Fraud detection - check for numerical inputs first
        if any(
            word in query_lower
            for word in ["pay_ratio", "svc_per_bene", "total_beneficiaries"]
        ) and any(word in query_lower for word in ["fraud", "risk"]):
            try:
                # Extract numerical values from query
                import re

                numbers = re.findall(r"\d+\.?\d*", query)
                if len(numbers) >= 3:
                    result = fraud_predict.invoke(
                        {
                            "pay_ratio": float(numbers[0]),
                            "svc_per_bene": float(numbers[1]),
                            "total_beneficiaries": float(numbers[2]),
                        }
                    )
                    return result
            except Exception as e:
                return {"output": f"Error in fraud prediction: {str(e)}"}

        # General fraud questions
        elif any(
            word in query_lower
            for word in ["fraud", "fraudulent", "billing patterns", "methodology"]
        ):
            import os

            if os.getenv("OPENAI_API_KEY"):
                try:
                    from langchain_openai import ChatOpenAI

                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                    response = llm.invoke(
                        f"Answer this healthcare fraud detection question: {query}. Be concise and factual."
                    )
                    return {
                        "output": f"**Fraud Detection Information**\n\n{response.content}\n\n*For specific fraud risk assessment, provide numerical metrics: pay_ratio, svc_per_bene, total_beneficiaries*"
                    }
                except Exception as e:
                    return {
                        "output": f"**Fraud Detection Information**\n\nI can help with fraud risk assessment using numerical metrics. Please provide:\n• **pay_ratio** - Payment to charge ratio\n• **svc_per_bene** - Services per beneficiary\n• **total_beneficiaries** - Number of beneficiaries\n\n*General fraud information unavailable: {str(e)}*"
                    }
            else:
                return {
                    "output": "**Fraud Detection Information**\n\nI can help with fraud risk assessment using numerical metrics. Please provide:\n• **pay_ratio** - Payment to charge ratio\n• **svc_per_bene** - Services per beneficiary\n• **total_beneficiaries** - Number of beneficiaries\n\n*Set OPENAI_API_KEY for general fraud information.*"
                }

        # Operational forecasting
        elif any(word in query_lower for word in ["forecast", "operational", "ops"]):
            try:
                # Check if ops_forecast model exists
                if Path("models/ops_forecast.joblib").exists():
                    result = ops_forecast.invoke({"query": query})
                    return result
                else:
                    return {
                        "output": "**Operational Forecasting**\n\nOperational forecasting model is not available on this instance due to resource constraints. However, I can help with:\n• **Fraud Risk Assessment** - Provide pay_ratio, svc_per_bene, total_beneficiaries\n• **Anomaly Detection** - Analyze unusual patterns\n• **Compliance Search** - Search CMS/HIPAA documents"
                    }
            except Exception as e:
                return {"output": f"Error in operational forecasting: {str(e)}"}

        # Anomaly detection
        elif any(
            word in query_lower
            for word in ["anomaly", "outlier", "unusual", "detect anomalies"]
        ):
            # Check if query has numerical inputs
            import re

            numbers = re.findall(r"[\d.]+", query)
            if len(numbers) >= 3:
                try:
                    result = ops_anomaly.invoke({"query": query})
                    return result
                except Exception as e:
                    return {"output": f"Error in anomaly detection: {str(e)}"}
            else:
                return {
                    "output": "**Anomaly Detection**\n\nPlease provide numerical values for anomaly detection:\n• **svc_per_bene** - Services per beneficiary\n• **total_beneficiaries** - Number of beneficiaries\n• **mean_payment** - Average payment amount\n\nExample: 'Detect anomalies for provider with svc_per_bene=22, total_beneficiaries=85, mean_payment=970.2'"
                }

        # Compliance search
        elif any(
            word in query_lower
            for word in ["compliance", "policy", "hipaa", "cms", "regulation"]
        ):
            try:
                result = compliance_search.invoke(query)
                return result
            except Exception as e:
                return {"output": f"Error in compliance search: {str(e)}"}

        # Business risk analysis
        elif any(
            word in query_lower
            for word in [
                "business risk",
                "hospital",
                "high pay_ratio",
                "unusual service patterns",
            ]
        ):
            import os

            if os.getenv("OPENAI_API_KEY"):
                try:
                    from langchain_openai import ChatOpenAI

                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                    response = llm.invoke(
                        f"Analyze this healthcare business risk scenario: {query}. Focus on operational and financial implications. Be concise and practical."
                    )
                    return {
                        "output": f"**Business Risk Analysis**\n\n{response.content}\n\n*For specific risk assessment, provide numerical metrics for fraud detection or operational forecasting.*"
                    }
                except Exception as e:
                    return {
                        "output": f"**Business Risk Analysis**\n\nI can help analyze business risks using specific metrics. Please provide:\n• **Fraud Risk Assessment** - pay_ratio, svc_per_bene, total_beneficiaries\n• **Operational Forecasting** - total_beneficiaries, mean_payment, mean_charge\n\n*General risk analysis unavailable: {str(e)}*"
                    }
            else:
                return {
                    "output": "**Business Risk Analysis**\n\nI can help analyze business risks using specific metrics. Please provide:\n• **Fraud Risk Assessment** - pay_ratio, svc_per_bene, total_beneficiaries\n• **Operational Forecasting** - total_beneficiaries, mean_payment, mean_charge\n\n*Set OPENAI_API_KEY for general risk analysis.*"
                }

        # General questions - use OpenAI fallback
        elif any(
            word in query_lower
            for word in [
                "what is",
                "explain",
                "how does",
                "general",
                "healthcare",
                "medical",
            ]
        ):
            import os

            if os.getenv("OPENAI_API_KEY"):
                try:
                    from langchain_openai import ChatOpenAI

                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                    response = llm.invoke(
                        f"Answer this healthcare question: {query}. Be concise and factual."
                    )
                    return {
                        "output": f"**General Healthcare Information**\n\n{response.content}\n\n*For specific predictions, provide numerical metrics for fraud detection, operational forecasting, or anomaly detection.*"
                    }
                except Exception as e:
                    return {
                        "output": f"**AI MedGuard (Local Mode)**\n\nI can help with specific predictions using your data. Please provide numerical metrics for:\n• **Fraud Risk Assessment**\n• **Operational Forecasting**\n• **Anomaly Detection**\n\n*General question fallback unavailable: {str(e)}*"
                    }
            else:
                return {
                    "output": "**AI MedGuard (Local Mode)**\n\nI can help with specific predictions using your data. Please provide numerical metrics for:\n• **Fraud Risk Assessment**\n• **Operational Forecasting**\n• **Anomaly Detection**\n\n*Set OPENAI_API_KEY environment variable for general healthcare questions.*"
                }

        # Default response
        import os

        fallback_info = ""
        if os.getenv("OPENAI_API_KEY"):
            fallback_info = "\n• **General Healthcare Questions** - Ask about medical concepts, regulations, or procedures\n\n*Local models + OpenAI fallback for comprehensive assistance.*"
        else:
            fallback_info = "\n\n*Running on local models and databases - set OPENAI_API_KEY for general question fallback.*"

        return {
            "output": f"**AI MedGuard (Local Mode)**\n\n"
            f"I can help you with:\n"
            f"• **Fraud Risk Assessment** - Provide pay_ratio, svc_per_bene, total_beneficiaries\n"
            f"• **Operational Forecasting** - Provide operational metrics\n"
            f"• **Anomaly Detection** - Analyze unusual patterns\n"
            f"• **Compliance Search** - Search CMS/HIPAA documents{fallback_info}"
        }


def _check_local_resources() -> bool:
    """
    Check if local resources are available and working.
    Returns True if local resources can be used, False otherwise.
    """
    try:
        # Check if essential models exist (ops_forecast is optional due to size)
        essential_models = [
            "models/fraud_baseline.joblib",
            "models/ops_anomaly.joblib",
        ]

        optional_models = [
            "models/ops_forecast.joblib",
        ]

        for model_path in essential_models:
            if not Path(model_path).exists():
                print(f"Warning: Essential model not found: {model_path}")
                return False

        for model_path in optional_models:
            if not Path(model_path).exists():
                print(
                    f"Info: Optional model not found: {model_path} (will skip forecasting)"
                )

        # Check if RAG database exists
        rag_path = "rag/vectorstore"
        if not Path(rag_path).exists():
            print(f"Warning: RAG database not found: {rag_path}")
            return False

        # Test if essential tools work
        test_result = fraud_predict.invoke(
            {"pay_ratio": 1.0, "svc_per_bene": 10, "total_beneficiaries": 50}
        )

        print("Success: Local resources available and working")
        return True

    except Exception as e:
        print(f"Warning: Local resources check failed: {e}")
        return False


def _create_local_agent() -> Optional["LocalAgent"]:
    """
    Create an agent that works with local resources only.
    This agent provides direct tool access without LLM routing.
    """
    try:
        # Create a simple local agent that routes queries to appropriate tools
        class LocalAgent:
            def __init__(self):
                self.tools = [
                    fraud_predict,
                    ops_forecast,
                    ops_anomaly,
                    compliance_search,
                ]

            def invoke(self, inputs):
                # Handle both string and dict inputs
                if isinstance(inputs, str):
                    query = inputs.lower()
                else:
                    query = inputs.get("input", "").lower()

                # Route queries to appropriate tools based on keywords
                # Fraud detection - check for numerical inputs first
                if any(
                    word in query
                    for word in ["pay_ratio", "svc_per_bene", "total_beneficiaries"]
                ) and any(word in query for word in ["fraud", "risk"]):
                    # Extract numbers from query
                    import re

                    numbers = re.findall(r"[\d.]+", query)
                    if len(numbers) >= 3:
                        try:
                            result = fraud_predict.invoke(
                                {
                                    "pay_ratio": float(numbers[0]),
                                    "svc_per_bene": float(numbers[1]),
                                    "total_beneficiaries": float(numbers[2]),
                                }
                            )
                            return {
                                "output": f"**Fraud Risk Analysis**\n\n{result}\n\n*Analysis based on your local fraud detection model.*"
                            }
                        except:
                            pass

                # General fraud questions
                elif any(
                    word in query
                    for word in [
                        "fraud",
                        "fraudulent",
                        "billing patterns",
                        "methodology",
                    ]
                ):
                    import os

                    if os.getenv("OPENAI_API_KEY"):
                        try:
                            from langchain_openai import ChatOpenAI

                            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                            response = llm.invoke(
                                f"Answer this healthcare fraud detection question: {query}. Be concise and factual."
                            )
                            return {
                                "output": f"**Fraud Detection Information**\n\n{response.content}\n\n*For specific fraud risk assessment, provide numerical metrics: pay_ratio, svc_per_bene, total_beneficiaries*"
                            }
                        except Exception as e:
                            return {
                                "output": f"**Fraud Detection Information**\n\nI can help with fraud risk assessment using numerical metrics. Please provide:\n• **pay_ratio** - Payment to charge ratio\n• **svc_per_bene** - Services per beneficiary\n• **total_beneficiaries** - Number of beneficiaries\n\n*General fraud information unavailable: {str(e)}*"
                            }
                    else:
                        return {
                            "output": "**Fraud Detection Information**\n\nI can help with fraud risk assessment using numerical metrics. Please provide:\n• **pay_ratio** - Payment to charge ratio\n• **svc_per_bene** - Services per beneficiary\n• **total_beneficiaries** - Number of beneficiaries\n\n*Set OPENAI_API_KEY for general fraud information.*"
                        }

                elif "forecast" in query or "operational" in query:
                    import re

                    numbers = re.findall(r"[\d.]+", query)
                    if len(numbers) >= 3:
                        try:
                            # Use simple operational forecasting function
                            def simple_operational_forecast(
                                total_beneficiaries, mean_payment, mean_charge
                            ):
                                """Simple operational forecasting based on basic healthcare metrics"""
                                total_revenue = total_beneficiaries * mean_payment
                                total_charges = total_beneficiaries * mean_charge
                                payment_ratio = (
                                    mean_payment / mean_charge if mean_charge > 0 else 0
                                )

                                # Simple forecasting logic
                                if payment_ratio > 0.6:
                                    risk_level = "LOW"
                                    efficiency_score = "HIGH"
                                    forecast = "Stable operations expected"
                                elif payment_ratio > 0.3:
                                    risk_level = "MEDIUM"
                                    efficiency_score = "MODERATE"
                                    forecast = "Monitor payment patterns closely"
                                else:
                                    risk_level = "HIGH"
                                    efficiency_score = "LOW"
                                    forecast = "Review billing practices and payment collection"

                                return f"""**Operational Forecast Results:**

**Provider Metrics:**
- Total Beneficiaries: {total_beneficiaries:,}
- Mean Payment: ${mean_payment:.2f}
- Mean Charge: ${mean_charge:.2f}
- Payment Ratio: {payment_ratio:.1%}

**Forecast Analysis:**
- Risk Level: {risk_level}
- Efficiency Score: {efficiency_score}
- Revenue Projection: ${total_revenue:,.2f}
- Charge Projection: ${total_charges:,.2f}

**Recommendation:** {forecast}

*Based on standard healthcare operational metrics and payment patterns.*"""

                            result = simple_operational_forecast(
                                float(numbers[0]),  # total_beneficiaries
                                float(numbers[1]),  # mean_payment
                                float(numbers[2]),  # mean_charge
                            )
                            return {"output": result}
                        except Exception as e:
                            return {
                                "output": f"**Operational Forecasting**\n\nError in operational forecasting: {str(e)}\n\n*Please provide valid numerical values for total_beneficiaries, mean_payment, and mean_charge.*"
                            }
                    else:
                        return {
                            "output": "**Operational Forecasting**\n\nPlease provide numerical values for:\n• **total_beneficiaries** - Number of beneficiaries\n• **mean_payment** - Average payment amount\n• **mean_charge** - Average charge amount\n\nExample: 'Forecast operational metrics for provider with total_beneficiaries=120, mean_payment=350.4, mean_charge=620.8'"
                        }

                elif any(
                    word in query
                    for word in ["anomaly", "outlier", "unusual", "detect anomalies"]
                ):
                    import re

                    numbers = re.findall(r"[\d.]+", query)
                    if len(numbers) >= 3:
                        try:
                            result = ops_anomaly.invoke(
                                {
                                    "svc_per_bene": float(numbers[0]),
                                    "total_beneficiaries": float(numbers[1]),
                                    "mean_payment": float(numbers[2]),
                                }
                            )
                            return {
                                "output": f"**Anomaly Detection**\n\n{result}\n\n*Analysis based on your local anomaly detection model.*"
                            }
                        except:
                            pass

                elif any(
                    word in query
                    for word in [
                        "cms",
                        "hipaa",
                        "compliance",
                        "policy",
                        "manual",
                        "regulation",
                        "resubmitting",
                        "denied",
                        "claims",
                    ]
                ):
                    # Extract search terms
                    search_terms = (
                        query.replace("what does", "")
                        .replace("summarize", "")
                        .replace("tell me about", "")
                        .strip()
                    )
                    try:
                        result = compliance_search.invoke({"query": search_terms})
                        return {
                            "output": f"**Compliance Search Results**\n\n{result}\n\n*Results from your local CMS/HIPAA document database.*"
                        }
                    except Exception as e:
                        return {
                            "output": f"**Compliance Search**\n\nCompliance search is temporarily unavailable due to resource constraints. However, I can help with:\n• **Fraud Risk Assessment** - Provide pay_ratio, svc_per_bene, total_beneficiaries\n• **Operational Forecasting** - Provide operational metrics\n• **Anomaly Detection** - Analyze unusual patterns\n\n*For general compliance questions, set OPENAI_API_KEY for AI-powered responses.*"
                        }

                # Business risk analysis
                elif any(
                    word in query
                    for word in [
                        "business risk",
                        "hospital",
                        "high pay_ratio",
                        "unusual service patterns",
                    ]
                ):
                    import os

                    if os.getenv("OPENAI_API_KEY"):
                        try:
                            from langchain_openai import ChatOpenAI

                            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                            response = llm.invoke(
                                f"Analyze this healthcare business risk scenario: {query}. Focus on operational and financial implications. Be concise and practical."
                            )
                            return {
                                "output": f"**Business Risk Analysis**\n\n{response.content}\n\n*For specific risk assessment, provide numerical metrics for fraud detection or operational forecasting.*"
                            }
                        except Exception as e:
                            return {
                                "output": f"**Business Risk Analysis**\n\nI can help analyze business risks using specific metrics. Please provide:\n• **Fraud Risk Assessment** - pay_ratio, svc_per_bene, total_beneficiaries\n• **Operational Forecasting** - total_beneficiaries, mean_payment, mean_charge\n\n*General risk analysis unavailable: {str(e)}*"
                            }
                    else:
                        return {
                            "output": "**Business Risk Analysis**\n\nI can help analyze business risks using specific metrics. Please provide:\n• **Fraud Risk Assessment** - pay_ratio, svc_per_bene, total_beneficiaries\n• **Operational Forecasting** - total_beneficiaries, mean_payment, mean_charge\n\n*Set OPENAI_API_KEY for general risk analysis.*"
                        }

                # General healthcare questions fallback
                elif any(
                    word in query
                    for word in [
                        "what is",
                        "explain",
                        "how does",
                        "general",
                        "healthcare",
                        "medical",
                    ]
                ):
                    # Use OpenAI fallback for general questions
                    import os

                    if os.getenv("OPENAI_API_KEY"):
                        try:
                            from langchain_openai import ChatOpenAI

                            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                            response = llm.invoke(
                                f"Answer this healthcare question: {query}. Be concise and factual."
                            )
                            return {
                                "output": f"**General Healthcare Information**\n\n{response.content}\n\n*For specific predictions, provide numerical metrics for fraud detection, operational forecasting, or anomaly detection.*"
                            }
                        except Exception as e:
                            return {
                                "output": f"**AI MedGuard (Local Mode)**\n\nI can help with specific predictions using your data. Please provide numerical metrics for:\n• **Fraud Risk Assessment**\n• **Operational Forecasting**\n• **Anomaly Detection**\n\n*General question fallback unavailable: {str(e)}*"
                            }
                    else:
                        return {
                            "output": "**AI MedGuard (Local Mode)**\n\nI can help with specific predictions using your data. Please provide numerical metrics for:\n• **Fraud Risk Assessment**\n• **Operational Forecasting**\n• **Anomaly Detection**\n\n*Set OPENAI_API_KEY environment variable for general healthcare questions.*"
                        }

                # Default response
                import os

                fallback_info = ""
                if os.getenv("OPENAI_API_KEY"):
                    fallback_info = "\n• **General Healthcare Questions** - Ask about medical concepts, regulations, or procedures\n\n*Local models + OpenAI fallback for comprehensive assistance.*"
                else:
                    fallback_info = "\n\n*Running on local models and databases - set OPENAI_API_KEY for general question fallback.*"

                return {
                    "output": f"**AI MedGuard (Local Mode)**\n\n"
                    f"I can help you with:\n"
                    f"• **Fraud Risk Assessment** - Provide pay_ratio, svc_per_bene, total_beneficiaries\n"
                    f"• **Operational Forecasting** - Provide operational metrics\n"
                    f"• **Anomaly Detection** - Analyze unusual patterns\n"
                    f"• **Compliance Search** - Search CMS/HIPAA documents{fallback_info}"
                }

        return LocalAgent()

    except Exception as e:
        print(f"Error: Failed to create local agent: {e}")
        return None


def _llm():
    # Force local mode - check if local resources are available
    if _check_local_resources():
        print("Using local resources (models + RAG database)")
        return "local"  # Special marker for local mode

    # If local resources not available, provide helpful error
    raise ValueError(
        "Local resources not available. Please ensure:\n"
        "1. ML models exist: models/fraud_baseline.joblib, models/ops_forecast.joblib, models/ops_anomaly.joblib\n"
        "2. RAG database exists: rag/vectorstore\n"
        "3. Run model training and RAG ingestion if needed"
    )


def build_agent() -> Union[AgentExecutor, LocalAgent, None]:
    """
    Build agent with priority: Local resources first, then OpenAI API fallback.
    """
    llm = _llm()

    # If using local mode, return local agent
    if llm == "local":
        local_agent = _create_local_agent()
        if local_agent:
            return local_agent
        else:
            raise ValueError("Failed to create local agent")

    # Otherwise, use OpenAI API with LangChain
    tools = [fraud_predict, ops_forecast, ops_anomaly, compliance_search]
    system = (
        "You are AI MedGuard, an assistant for hospital analytics. "
        "When asked about risk, call the appropriate tools and summarize clearly. "
        "If the query mentions policy or HIPAA/CMS, use compliance_search. "
        "When given features, call fraud_predict or ops_forecast/ops_anomaly. "
        "CRITICAL: For ML tools, ALWAYS pass parameters individually. "
        "Example: fraud_predict(pay_ratio=1.25, svc_per_bene=18, total_beneficiaries=42). "
        "NEVER use features_json or any dictionary parameters. "
        "Always explain briefly what the numbers mean for a hospital QA leader."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


# Quick CLI
if __name__ == "__main__":
    agent = build_agent()
    if agent:
        if hasattr(agent, "invoke") and callable(getattr(agent, "invoke")):
            # Check if it's LocalAgent (expects string) or AgentExecutor (expects dict)
            if isinstance(agent, LocalAgent):
                resp = agent.invoke(
                    "Search CMS HIPAA privacy summary about claims submission"
                )
            else:
                resp = agent.invoke(
                    {
                        "input": "Search CMS HIPAA privacy summary about claims submission"
                    }
                )
            print(resp["output"] if isinstance(resp, dict) else resp)
        else:
            print("Agent does not have invoke method")
    else:
        print("Failed to build agent")
