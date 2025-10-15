from __future__ import annotations
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.tools import tool
from rag.query_rag import query_top_k


@tool("compliance_search")
def compliance_search(query: str) -> str:
    """
    Search CMS and HIPAA compliance documents for relevant policy information.

    Use this tool to find information about:
    - Claims submission and billing requirements
    - HIPAA privacy and security rules
    - Medicare/Medicaid regulations
    - Healthcare compliance policies

    Args:
        query: Natural language question about compliance (e.g., "What does CMS say about claims submission?")

    Returns:
        Relevant policy excerpts with source citations, or general information if no documents found
    """
    docs = query_top_k(query, k=4)
    out = []
    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source", "unknown").split("/")[-1]  # Just filename
        page = meta.get("page", "?")
        content = d.page_content[:400]  # Limit length
        out.append(f"Source: {source} (Page {page})\n{content}...\n")

    if not out:
        # FALLBACK: Use OpenAI for general healthcare compliance questions
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                response = llm.invoke(
                    f"Answer this healthcare compliance question based on general knowledge: {query}. Focus on CMS, HIPAA, Medicare/Medicaid, and healthcare billing regulations. Be concise and factual."
                )
                return f"**General Healthcare Compliance Information**\n\n{response.content}\n\n*Note: This is general information, not from specific CMS/HIPAA documents in our database.*"
            except Exception as e:
                return f"No relevant compliance documents found for this query. (OpenAI fallback unavailable: {str(e)})"
        else:
            return "No relevant compliance documents found for this query. Set OPENAI_API_KEY environment variable for general information fallback."

    return "\n".join(out)
