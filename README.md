# MedExplain

**Medical Report Analysis System**

A healthcare information system that provides automated interpretation assistance for medical laboratory reports and imaging studies. Designed to help patients understand their medical documentation while emphasizing the importance of professional medical consultation.

---

## Important Notice

**This system does not provide medical diagnoses.**

All analysis results are for educational and informational purposes only. Users must consult qualified healthcare professionals before making any medical decisions. This software is not a substitute for professional medical advice, diagnosis, or treatment.

---

## Overview

MedExplain processes medical reports and provides structured interpretations in accessible language. The system is designed with healthcare compliance in mind, incorporating:

- Mandatory safety disclaimers on all outputs
- Language filtering to prevent diagnostic claims
- Confidence scoring for transparency
- Professional clinical terminology

### Key Features

- **Document Processing**: PDF and text file analysis with content extraction
- **Medical Imaging Support**: Processing pipeline for radiograph and imaging files
- **Automated Interpretation**: Rule-based and LLM-assisted content analysis
- **Safety Compliance**: Built-in safeguards against inappropriate medical claims
- **Report Generation**: Professional PDF output with proper disclaimers

---

## Architecture

```
medexplain/
├── app/
│   ├── api/                 # REST API endpoints
│   │   ├── routes.py        # Request handlers
│   │   └── middleware.py    # Request processing middleware
│   ├── core/                # Core processing modules
│   │   ├── llm_engine.py    # Language model integration
│   │   ├── vision_model.py  # Image processing pipeline
│   │   ├── pdf_extractor.py # Document parsing
│   │   └── text_processor.py# Text analysis utilities
│   ├── services/            # Business logic
│   │   ├── report_analyzer.py   # Analysis orchestration
│   │   ├── safety_checker.py    # Compliance validation
│   │   └── report_generator.py  # Output generation
│   └── models/              # Data models
├── frontend/                # Web interface
│   ├── index.html          # Application interface
│   ├── styles.css          # Styling
│   └── app.js              # Client application
└── tests/                   # Test suite
```

---

## Technical Stack

| Component      | Technology                          |
|----------------|-------------------------------------|
| Backend        | Python 3.9+, FastAPI                |
| Document Processing | pdfplumber, Pillow             |
| Machine Learning | PyTorch, torchvision             |
| LLM Integration  | Google Gemini API (optional)      |
| Report Generation | Jinja2, WeasyPrint              |
| Web Interface   | HTML5, CSS3, JavaScript (Vanilla)  |
| Deployment      | Docker, Docker Compose             |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/medexplain.git
cd medexplain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Configuration

Create a `.env` file with the following variables:

```env
# Optional: External LLM provider
GEMINI_API_KEY=your_api_key_here

# Application settings
DEBUG=false
LOG_LEVEL=INFO
PORT=8000

# Processing limits
MAX_FILE_SIZE_MB=10
RATE_LIMIT_PER_MINUTE=30
```

---

## Usage

### Starting the Server

```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Web Interface

Access the application at `http://localhost:8000`

### API Endpoints

| Endpoint           | Method | Description                    |
|-------------------|--------|--------------------------------|
| `/health`         | GET    | System health check            |
| `/upload-report`  | POST   | Upload laboratory report       |
| `/upload-xray`    | POST   | Upload medical imaging file    |
| `/generate-report`| POST   | Generate analysis report       |
| `/create-pdf`     | POST   | Generate PDF document          |
| `/download-pdf/{id}` | GET | Download generated PDF        |

---

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Testing

```bash
# Run test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

---

## Safety Guidelines

This system implements multiple safety measures:

1. **No Diagnostic Claims**: All outputs explicitly state they are not medical diagnoses
2. **Mandatory Disclaimers**: Every analysis includes consultation reminders
3. **Language Filtering**: Removes alarming or inappropriate terminology
4. **Confidence Scoring**: Transparent indication of analysis reliability
5. **Professional Review Flags**: Identifies content requiring clinical review

### Required Disclaimers

All system outputs include:

- "This analysis does not constitute a medical diagnosis"
- "Consult a qualified healthcare provider before making medical decisions"
- "Information provided is for educational purposes only"

---

## Compliance Considerations

When deploying this system:

- Ensure compliance with applicable healthcare regulations (HIPAA, GDPR, etc.)
- Implement appropriate access controls and audit logging
- Review all outputs before clinical use
- Maintain documentation of system limitations

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Disclaimer

This software is provided "as is" without warranty of any kind. The developers and contributors are not liable for any damages arising from the use of this software. This system is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions regarding medical conditions.

---

**MedExplain** - Medical Report Analysis System  
Version 1.0.0
