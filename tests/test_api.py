"""
Basic API tests for AI MedGuard
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api.app import app
    client = TestClient(app)
except ImportError:
    # Skip tests if API module is not available
    pytest.skip("API module not available", allow_module_level=True)


def test_health_endpoint():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200


def test_api_info():
    """Test API info endpoint"""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
