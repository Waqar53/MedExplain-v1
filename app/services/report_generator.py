"""
PDF report generator for MedExplain AI.

Creates professional PDF reports from analysis results.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Template

# Try to import WeasyPrint, make it optional
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    WEASYPRINT_AVAILABLE = False
    HTML = None
    CSS = None

from app.config import settings
from app.models.schemas import PatientReport, AnalysisResult
from app.utils.logger import get_logger

logger = get_logger("report_generator")


class ReportGenerator:
    """
    Generates professional PDF reports for patients.
    
    Uses HTML templates and WeasyPrint for PDF conversion.
    All reports include:
    - Patient-friendly summary
    - What results indicate
    - Common next steps
    - Confidence level
    - Safety disclaimers
    """
    
    # CSS styles for PDF reports
    REPORT_CSS = """
    @page {
        size: A4;
        margin: 2cm;
        @top-right {
            content: "MedExplain AI Report";
            font-size: 9pt;
            color: #666;
        }
        @bottom-center {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 9pt;
            color: #666;
        }
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #333;
    }
    
    .header {
        text-align: center;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }
    
    .header h1 {
        color: #0066cc;
        font-size: 24pt;
        margin: 0;
    }
    
    .header .subtitle {
        color: #666;
        font-size: 12pt;
        margin-top: 5px;
    }
    
    .info-box {
        background: #f5f5f5;
        border-left: 4px solid #0066cc;
        padding: 15px;
        margin: 20px 0;
    }
    
    .info-box.warning {
        border-left-color: #ff9800;
        background: #fff8e1;
    }
    
    .info-box.danger {
        border-left-color: #f44336;
        background: #ffebee;
    }
    
    h2 {
        color: #0066cc;
        font-size: 14pt;
        margin-top: 25px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
    }
    
    .summary {
        font-size: 12pt;
        background: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    .findings-list {
        list-style-type: none;
        padding: 0;
    }
    
    .findings-list li {
        padding: 8px 0;
        border-bottom: 1px solid #eee;
    }
    
    .findings-list li:before {
        content: "•";
        color: #0066cc;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .next-steps {
        background: #e8f5e9;
        padding: 15px;
        border-radius: 5px;
    }
    
    .next-steps ol {
        margin: 0;
        padding-left: 20px;
    }
    
    .next-steps li {
        padding: 5px 0;
    }
    
    .confidence-indicator {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 15px;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 10pt;
    }
    
    .confidence-high {
        background: #c8e6c9;
        color: #2e7d32;
    }
    
    .confidence-medium {
        background: #fff9c4;
        color: #f57f17;
    }
    
    .confidence-low {
        background: #ffcdd2;
        color: #c62828;
    }
    
    .disclaimer {
        margin-top: 40px;
        padding: 20px;
        background: #fff3e0;
        border: 1px solid #ffcc80;
        border-radius: 5px;
    }
    
    .disclaimer h3 {
        color: #e65100;
        margin-top: 0;
    }
    
    .disclaimer p {
        margin: 10px 0;
        font-size: 10pt;
    }
    
    .footer {
        margin-top: 40px;
        text-align: center;
        font-size: 9pt;
        color: #999;
        border-top: 1px solid #ddd;
        padding-top: 20px;
    }
    
    .risk-indicator {
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 10pt;
    }
    
    .risk-low { background: #c8e6c9; color: #2e7d32; }
    .risk-medium { background: #fff9c4; color: #f57f17; }
    .risk-high { background: #ffcdd2; color: #c62828; }
    .risk-unknown { background: #e0e0e0; color: #616161; }
    """
    
    def __init__(self):
        self.output_dir = settings.output_path
        self._setup_template_env()
    
    def _setup_template_env(self) -> None:
        """Setup Jinja2 template environment."""
        # Use inline template since we're keeping it simple
        self.template_html = self._get_html_template()
    
    def _get_html_template(self) -> str:
        """Get the HTML template for reports."""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MedExplain AI Report</title>
</head>
<body>
    <div class="header">
        <h1>📋 Medical Report Explanation</h1>
        <div class="subtitle">Generated by MedExplain AI</div>
        <div class="subtitle">{{ generated_date }}</div>
        {% if patient_name %}
        <div class="subtitle"><strong>Patient:</strong> {{ patient_name }}</div>
        {% endif %}
    </div>
    
    <div class="info-box warning">
        <strong>⚠️ Important Notice:</strong> This is NOT a medical diagnosis. 
        This report provides educational information only. Please consult your 
        healthcare provider for medical advice.
    </div>
    
    <h2>📝 Summary</h2>
    <div class="summary">
        {{ summary }}
    </div>
    
    <h2>📊 Analysis Details</h2>
    <p>
        <strong>Confidence Level:</strong> 
        <span class="confidence-indicator confidence-{{ confidence }}">{{ confidence }}</span>
    </p>
    <p>
        <strong>Risk Category:</strong>
        <span class="risk-indicator risk-{{ risk_level }}">{{ risk_level }}</span>
        <em>(Informational only)</em>
    </p>
    
    <h2>💡 What This Generally Means</h2>
    <p>{{ what_this_means }}</p>
    
    {% if key_findings %}
    <h2>🔍 Key Findings</h2>
    <ul class="findings-list">
        {% for finding in key_findings %}
        <li>{{ finding }}</li>
        {% endfor %}
    </ul>
    {% endif %}
    
    <h2>📋 Common Next Steps</h2>
    <div class="next-steps">
        <p>Healthcare providers commonly recommend:</p>
        <ol>
            {% for step in common_next_steps %}
            <li>{{ step }}</li>
            {% endfor %}
        </ol>
        <p><em>These are general suggestions, not specific recommendations for your case.</em></p>
    </div>
    
    <div class="disclaimer">
        <h3>⚠️ Important Disclaimer</h3>
        <p><strong>{{ disclaimer_main }}</strong></p>
        <p>{{ disclaimer_consultation }}</p>
        {% if confidence_note %}
        <p>ℹ️ {{ confidence_note }}</p>
        {% endif %}
        <p>
            This analysis was generated by an AI system and has not been reviewed 
            by a medical professional. The information provided is for educational 
            purposes only and should not be used as a substitute for professional 
            medical advice, diagnosis, or treatment.
        </p>
    </div>
    
    <div class="footer">
        <p>Report ID: {{ report_id }}</p>
        <p>Generated: {{ generated_date }} | MedExplain AI v{{ version }}</p>
        <p>This report was created to help you understand your medical information better.</p>
        <p>Always discuss your results with your healthcare provider.</p>
    </div>
</body>
</html>
"""
    
    def generate_pdf(
        self,
        report: PatientReport,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Generate a PDF report.
        
        Args:
            report: PatientReport with analysis results
            output_filename: Optional custom filename
            
        Returns:
            Path to generated PDF file (or HTML if WeasyPrint unavailable)
        """
        logger.info("Generating PDF report", report_id=report.report_id)
        
        # Check if WeasyPrint is available
        if not WEASYPRINT_AVAILABLE:
            logger.warning("WeasyPrint not available, generating HTML instead")
            # Generate HTML file as fallback
            html_content = self.generate_html(report)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_filename = f"medexplain_report_{timestamp}_{report.report_id[:8]}.html"
            output_path = self.output_dir / html_filename
            output_path.write_text(html_content)
            logger.info("HTML report generated (PDF unavailable)", path=str(output_path))
            return output_path
        
        # Prepare template data
        template_data = self._prepare_template_data(report)
        
        # Render HTML from template
        template = Template(self.template_html)
        html_content = template.render(**template_data)
        
        # Generate PDF
        if output_filename:
            pdf_filename = output_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"medexplain_report_{timestamp}_{report.report_id[:8]}.pdf"
        
        output_path = self.output_dir / pdf_filename
        
        # Create PDF with WeasyPrint
        html = HTML(string=html_content)
        css = CSS(string=self.REPORT_CSS)
        html.write_pdf(output_path, stylesheets=[css])
        
        logger.info(
            "PDF report generated",
            report_id=report.report_id,
            path=str(output_path)
        )
        
        return output_path
    
    def _prepare_template_data(self, report: PatientReport) -> dict:
        """Prepare data for template rendering."""
        analysis = report.analysis
        
        return {
            "report_id": report.report_id,
            "generated_date": report.generated_at.strftime("%B %d, %Y at %I:%M %p"),
            "patient_name": report.patient_name,
            "version": settings.app_version,
            
            # Analysis content
            "summary": analysis.explanation.summary,
            "what_this_means": analysis.explanation.what_this_means,
            "key_findings": analysis.explanation.key_findings,
            "common_next_steps": analysis.explanation.common_next_steps,
            
            # Confidence and risk
            "confidence": analysis.confidence.value,
            "confidence_score": f"{analysis.confidence_score:.0%}",
            "risk_level": analysis.risk_level.value,
            
            # Disclaimers
            "disclaimer_main": analysis.disclaimer.main_disclaimer,
            "disclaimer_consultation": analysis.disclaimer.consultation_reminder,
            "confidence_note": analysis.disclaimer.confidence_note
        }
    
    def generate_html(self, report: PatientReport) -> str:
        """
        Generate HTML report (without PDF conversion).
        
        Useful for email or web display.
        
        Args:
            report: PatientReport with analysis results
            
        Returns:
            HTML string
        """
        from jinja2 import Template
        template = Template(self.template_html)
        template_data = self._prepare_template_data(report)
        
        html_content = template.render(**template_data)
        
        # Add CSS inline for standalone HTML
        styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MedExplain AI Report</title>
    <style>{self.REPORT_CSS}</style>
</head>
<body>
{html_content.split('<body>')[1].split('</body>')[0]}
</body>
</html>
"""
        return styled_html


# Lazy-loaded singleton
_report_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Get or create report generator singleton."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator
