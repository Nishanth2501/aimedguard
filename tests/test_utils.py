"""
Basic utility tests for AI MedGuard
"""
import pytest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_utils_import():
    """Test that utils module can be imported"""
    try:
        from utils import metrics
        assert metrics is not None
    except ImportError:
        pytest.skip("Utils module not available")


def test_project_structure():
    """Test basic project structure"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check that key directories exist
    expected_dirs = ['api', 'agent', 'data', 'ml', 'models', 'rag', 'ui']
    for dir_name in expected_dirs:
        dir_path = os.path.join(project_root, dir_name)
        assert os.path.exists(dir_path), f"Directory {dir_name} should exist"


def test_requirements_file():
    """Test that requirements.txt exists and has content"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    requirements_path = os.path.join(project_root, 'requirements.txt')
    
    assert os.path.exists(requirements_path), "requirements.txt should exist"
    
    with open(requirements_path, 'r') as f:
        content = f.read().strip()
        assert len(content) > 0, "requirements.txt should not be empty"
        assert 'fastapi' in content.lower(), "requirements.txt should contain fastapi"
        assert 'streamlit' in content.lower(), "requirements.txt should contain streamlit"
