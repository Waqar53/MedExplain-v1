"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Health response should have correct structure."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"


class TestUploadEndpoints:
    """Test file upload endpoints."""
    
    def test_upload_empty_file_rejected(self, client):
        """Empty files should be rejected."""
        response = client.post(
            "/upload-report",
            files={"file": ("test.txt", b"", "text/plain")}
        )
        assert response.status_code == 400
    
    def test_upload_text_file_accepted(self, client):
        """Valid text files should be accepted."""
        content = b"Patient: John Doe\nGlucose: 95 mg/dL\nResult: Normal"
        response = client.post(
            "/upload-report",
            files={"file": ("report.txt", content, "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["file_type"] == "text"
    
    def test_upload_with_tenant_id(self, client):
        """Uploads with tenant ID should work."""
        content = b"Test report content"
        response = client.post(
            "/upload-report",
            files={"file": ("report.txt", content, "text/plain")},
            headers={"X-Tenant-ID": "clinic-001"}
        )
        
        assert response.status_code == 200


class TestGenerateReport:
    """Test report generation endpoint."""
    
    def test_generate_invalid_session(self, client):
        """Invalid session ID should return 404."""
        response = client.post(
            "/generate-report",
            json={"session_id": "invalid-session-id"}
        )
        assert response.status_code == 404
    
    def test_generate_requires_session_id(self, client):
        """Request without session_id should fail."""
        response = client.post(
            "/generate-report",
            json={}
        )
        assert response.status_code == 422  # Validation error


class TestRateLimiting:
    """Test rate limiting."""
    
    def test_health_not_rate_limited(self, client):
        """Health endpoint should work repeatedly."""
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200
