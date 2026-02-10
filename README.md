# Contract Compliance Agent

**AI-Powered Contract Analysis Using Local LLMs**

A local-first AI agent that analyzes contracts against configurable compliance rules and generates quantitative compliance scores based on established academic and industry frameworks.

> ⚠️ **Disclaimer**: This is a prototype implementation designed for educational and demonstration purposes. It is NOT intended for production use or as a substitute for professional legal review. The compliance scores and analysis provided should be validated by qualified legal professionals before any business decisions.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Scoring Methodology](#scoring-methodology)
7. [API Reference](#api-reference)
8. [Testing](#testing)
9. [Limitations](#limitations)
10. [References](#references)

---

## Features

- **Local-First Architecture**: Runs entirely on your machine using Ollama and open-source LLMs — no data leaves your infrastructure
- **Quantitative Compliance Scoring**: Based on CUAD benchmark methodology and ISO 37301/37302 frameworks
- **Multi-Format Support**: Processes PDF, DOCX, and TXT contract files
- **Multi-Clause Analysis**: Evaluates contracts against 15+ critical clause categories
- **Risk Categorization**: Traffic-light system (Green/Yellow/Red) with ISO maturity level mapping
- **Configurable Rules**: YAML-based compliance rules adaptable to organizational requirements
- **Multiple Output Formats**: JSON, Markdown, and formatted terminal reports
- **Full Test Suite**: Unit and integration tests for all components

---

## Architecture

The agent follows a modular pipeline architecture separating concerns for maintainability and extensibility.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Interface                                   │
│                              (main.py)                                       │
│         Commands: analyze | status | list-rules | version                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ComplianceAgent                                    │
│                           (agent/core.py)                                    │
│                                                                              │
│  • Orchestrates the full analysis pipeline                                  │
│  • Manages configuration and rule loading                                   │
│  • Provides context manager for resource cleanup                            │
│  • Exposes high-level analyze() method                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  DocumentLoader   │   │  ContractAnalyzer │   │  ReportGenerator  │
│  (tools/)         │   │  (agent/)         │   │  (agent/)         │
│                   │   │                   │   │                   │
│ • load_document() │   │ • analyze()       │   │ • to_markdown()   │
│ • detect_format() │   │ • _extract()      │   │ • to_json()       │
│ • extract_text()  │   │ • _score()        │   │ • to_terminal()   │
│                   │   │ • _recommend()    │   │ • to_pdf()        │
│ Supports:         │   │                   │   │                   │
│ • PDF (pypdf)     │   │                   │   │                   │
│ • DOCX (docx)     │   │                   │   │                   │
│ • TXT (native)    │   │                   │   │                   │
└───────────────────┘   └───────────────────┘   └───────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
          ┌───────────────┐ ┌─────────────┐ ┌───────────────┐
          │ClauseExtractor│ │  Scorer     │ │ Recommender   │
          │(agent/)       │ │ (agent/)    │ │ (agent/)      │
          │               │ │             │ │               │
          │• extract()    │ │• presence() │ │• generate()   │
          │• prefilter()  │ │• similarity()│ │• prioritize() │
          │• parse_json() │ │• complete() │ │               │
          └───────────────┘ └─────────────┘ └───────────────┘
                    │
                    ▼
          ┌───────────────────┐
          │   OllamaClient    │
          │   (tools/)        │
          │                   │
          │ • generate()      │
          │ • check_status()  │
          │ • list_models()   │
          │                   │
          │ Retry logic:      │
          │ • 3 attempts      │
          │ • Exp. backoff    │
          └───────────────────┘
                    │
                    ▼
          ┌───────────────────┐
          │   Ollama Server   │
          │   (localhost)     │
          │                   │
          │ Models:           │
          │ • llama3.1:8b     │
          │ • qwen2.5:7b      │
          │ • phi3:mini       │
          └───────────────────┘
```

### Component Responsibilities

#### `ComplianceAgent` (agent/core.py)
The main orchestrator that coordinates all components. Implements context manager protocol for proper resource management.

```python
with ComplianceAgent(model="llama3.1:8b", verbose=True) as agent:
    result = agent.analyze("contract.pdf")
```

#### `ContractAnalyzer` (agent/analyzer.py)
Executes the analysis pipeline for a single contract:
1. Iterates through configured clause rules
2. Calls ClauseExtractor for each clause
3. Calculates presence, similarity, and completeness scores
4. Aggregates into document-level compliance score
5. Generates prioritized recommendations

#### `ClauseExtractor` (agent/extractor.py)
Extracts clause information using a two-stage approach:
1. **Keyword Pre-Filter**: Fast heuristic to identify likely clause locations
2. **LLM Extraction**: Structured extraction with JSON output parsing

#### `ComplianceScorer` (agent/scorer.py)
Implements the scoring methodology:
- Clause-level composite scoring
- Weighted document aggregation
- Risk level classification
- ISO maturity mapping

#### `OllamaClient` (tools/llm_client.py)
Handles all LLM interactions:
- Connection management
- Request/response handling
- Retry logic with exponential backoff
- Model availability checking

### Data Flow

```
Contract File
     │
     ▼
┌─────────────┐
│ Load & Parse │ ──► DocumentMetadata (filename, size, format, pages)
└─────────────┘
     │
     ▼
Contract Text (string)
     │
     ▼
┌─────────────────────────────────────────────┐
│         For each ClauseRule:                │
│  ┌─────────────┐    ┌──────────────────┐   │
│  │  Pre-filter │ ─► │  LLM Extraction  │   │
│  └─────────────┘    └──────────────────┘   │
│         │                    │              │
│         ▼                    ▼              │
│    Candidate         ClauseExtraction       │
│    Sections          (text, elements,       │
│                       confidence)           │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│              Score Calculation               │
│                                              │
│  presence_score = f(status, confidence)     │
│  similarity_score = LLM_eval(text, rule)    │
│  completeness_score = found / (found+miss)  │
│                                              │
│  clause_score = 0.3×P + 0.4×S + 0.3×C       │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│           Document Aggregation               │
│                                              │
│  doc_score = Σ(clause×weight) / Σ(weight)   │
│  risk_level = classify(doc_score)           │
│  maturity = map_to_iso(doc_score)           │
└─────────────────────────────────────────────┘
     │
     ▼
AnalysisResult
     │
     ▼
┌─────────────────┐
│ Report Output   │ ──► Markdown / JSON / Terminal / PDF
└─────────────────┘
```

---

## Installation

### What You Need

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10/11, macOS 12+, or Linux |
| **Python** | Version 3.10 or higher |
| **RAM** | Minimum 8GB, recommended 16GB |
| **Storage** | 10GB free for models |
| **GPU (Optional)** | NVIDIA GPU with 8GB+ VRAM for faster processing |
| **Ollama** | installed and running

### Step 1: Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
```

### Step 2: Pull a Model

```bash
# Recommended for most systems
ollama pull llama3.1:8b

# Alternative options
ollama pull qwen2.5:7b    # Better multilingual support
ollama pull phi3:mini      # For resource-constrained systems
```

### Step 3: Setup the Project

```bash
# Navigate to the project directory
cd compliance-agent

# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows CMD)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Check Ollama status
python main.py status

# Run a test analysis
python main.py analyze -c samples/nda_sample.txt -v -m qwen2.5:3b
```

---

## Usage

### Command Line Interface

The agent provides a CLI with multiple commands:

```bash
# Analyze a contract
python main.py analyze --contract path/to/contract.pdf

# Short form with options
python main.py analyze -c contract.docx -o report.md -v -m llama3.1:8b

# Check Ollama status
python main.py status

# List available compliance rules
python main.py list-rules

# Show version
python main.py version
```

### CLI Options for `analyze`

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--contract` | `-c` | Path to contract file (required) | — |
| `--output` | `-o` | Output file path | Terminal |
| `--model` | `-m` | Ollama model to use | `llama3.1:8b` |
| `--format` | `-f` | Output format: markdown, json, text | `markdown` |
| `--rules` | `-r` | Custom rules YAML file | Default rules |
| `--type` | `-t` | Contract type hint | `general` |
| `--verbose` | `-v` | Enable verbose output | `False` |

### Python API

```python
from agent import ComplianceAgent

# Basic usage
with ComplianceAgent(model="qwen2.5:3b") as agent:
    result = agent.analyze("contract.pdf")
    print(f"Score: {result.compliance_score.overall_score:.2f}")
    print(f"Risk: {result.compliance_score.risk_level.value}")

# With custom configuration
from config import load_rules

rules = load_rules("my_rules.yaml")
with ComplianceAgent(model="llama3.1:8b", rules=rules, verbose=True) as agent:
    result = agent.analyze(
        "contract.docx",
        contract_type=ContractType.SERVICE_AGREEMENT
    )
    agent.save_report(result, "report.md", format="markdown")
```

### Batch Processing

```python
from pathlib import Path
from agent import ComplianceAgent

with ComplianceAgent() as agent:
    results = []
    for contract in Path("contracts/").glob("*.pdf"):
        result = agent.analyze(contract)
        results.append({
            "file": contract.name,
            "score": result.compliance_score.overall_score,
            "risk": result.compliance_score.risk_level.value
        })
    
    # Sort by risk level
    high_risk = [r for r in results if r["risk"] == "high"]
    print(f"Found {len(high_risk)} high-risk contracts")
```

---

## Configuration

### Compliance Rules (`config/default_rules.yaml`)

Rules define what clauses to look for and how to evaluate them:

```yaml
clauses:
  - id: "termination"
    name: "Termination Rights"
    description: "Conditions under which parties can terminate the agreement"
    required: true
    risk_if_missing: "high"
    keywords:
      - "termination"
      - "terminate"
      - "cancellation"
      - "end of agreement"
    expected_elements:
      - "termination for cause"
      - "termination for convenience"
      - "notice period"
      - "effect of termination"

  - id: "liability"
    name: "Limitation of Liability"
    description: "Caps on liability and damage exclusions"
    required: true
    risk_if_missing: "high"
    keywords:
      - "limitation of liability"
      - "liability cap"
      - "damages"
      - "liable"
    expected_elements:
      - "aggregate cap"
      - "exclusion of consequential damages"
      - "carve-outs"
```

### Clause Weights (`config/clause_weights.yaml`)

Weights determine relative importance in the overall score:

```yaml
weights:
  termination: 0.95
  ip_assignment: 0.90
  liability: 0.90
  indemnification: 0.85
  confidentiality: 0.80
  governing_law: 0.75
  payment_terms: 0.70
  warranties: 0.65
  force_majeure: 0.60
  assignment: 0.55
```

### Settings (`config/settings.py`)

Global configuration options:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120  # seconds
DEFAULT_MODEL = "llama3.1:8b"
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# Scoring thresholds
RISK_THRESHOLD_HIGH = 0.40
RISK_THRESHOLD_MEDIUM = 0.70

# Output settings
DEFAULT_OUTPUT_FORMAT = "markdown"
```

---

## Scoring Methodology

### Academic Foundations

The scoring system is grounded in peer-reviewed research:

- **CUAD (NeurIPS 2021)**: Contract Understanding Atticus Dataset with 41 clause categories
- **LexGLUE (ACL 2022)**: Legal language understanding benchmark
- **ContractNLI (EMNLP 2021)**: Document-level inference for contracts

### Industry Frameworks

- **ISO 37301:2021**: Compliance management systems requirements
- **ISO 37302:2025**: Five-level maturity model
- **NIST CSF 2.0**: Four-tier implementation model

### Scoring Formula

**Clause-Level Score:**
```
clause_score = (presence × 0.3) + (similarity × 0.4) + (completeness × 0.3)
```

Where:
- `presence`: Detection confidence (0-1)
- `similarity`: Language quality assessment (0-1)
- `completeness`: found_elements / total_expected (0-1)

**Document-Level Score:**
```
doc_score = Σ(clause_score × clause_weight) / Σ(clause_weight)
```

**Risk Classification:**
| Score | Level | ISO 37302 Maturity |
|-------|-------|-------------------|
| ≥ 0.70 | Low | Level 4-5 (Advanced/Optimized) |
| 0.40-0.69 | Medium | Level 2-3 (Developing/Established) |
| < 0.40 | High | Level 1 (Basic) |

---

## Project Structure

```
compliance-agent/
├── agent/
│   ├── __init__.py           # Package exports, version
│   ├── core.py               # ComplianceAgent main class
│   ├── analyzer.py           # ContractAnalyzer pipeline
│   ├── extractor.py          # ClauseExtractor + KeywordPreFilter
│   ├── scorer.py             # ComplianceScorer + risk mapping
│   └── reporter.py           # ReportGenerator (md/json/terminal)
├── config/
│   ├── __init__.py           # Config loading utilities
│   ├── settings.py           # Global settings
│   ├── default_rules.yaml    # Default clause rules
│   └── clause_weights.yaml   # Clause importance weights
├── models/
│   ├── __init__.py           # Model exports
│   └── schemas.py            # Pydantic models (20+ types)
├── tools/
│   ├── __init__.py           # Tool exports
│   ├── document_loader.py    # PDF/DOCX/TXT parsing
│   └── llm_client.py         # OllamaClient with retry logic
├── tests/
│   ├── __init__.py
│   ├── test_scorer.py        # Scorer unit tests
│   └── test_integration.py   # End-to-end tests
├── samples/
│   ├── nda_sample.txt        # Sample NDA (complete)
│   └── service_agreement.txt # Sample service agreement (incomplete)
├── main.py                   # CLI entry point (Typer)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## Testing

### Run All Tests

```bash
# Basic test run
pytest tests/ -v

# With coverage report
pytest tests/ --cov=agent --cov-report=html

# Run only unit tests (no Ollama required)
pytest tests/test_scorer.py -v

# Run integration tests (requires Ollama)
pytest tests/test_integration.py -v
```

### Test Coverage Targets

| Module | Target | Notes |
|--------|--------|-------|
| `scorer.py` | 90%+ | Pure logic, fully testable |
| `extractor.py` | 80%+ | Mocked LLM responses |
| `analyzer.py` | 75%+ | Integration with mocks |
| `core.py` | 70%+ | Orchestration logic |

---

## Limitations

1. **Prototype Status**: Designed for demonstration and learning, not production
2. **Model Dependent**: Analysis quality varies with LLM capability
3. **English Optimized**: Best performance with English-language contracts
4. **Text-Based PDFs**: No OCR for scanned documents
5. **No Legal Advice**: Cannot replace professional legal review

---

## References

1. Hendrycks, D., et al. (2021). "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review." *NeurIPS Datasets and Benchmarks*.

2. Chalkidis, I., et al. (2022). "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English." *ACL*.

3. ISO 37301:2021. "Compliance management systems — Requirements with guidance for use."

4. ISO 37302:2025. "Compliance management systems — Guidelines on managing compliance and supporting ethics."

5. Koreeda, Y., & Manning, C. D. (2021). "ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts." *EMNLP*.

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Author

**Luigi Simeone** is an AI Consultant and Chief Scientist specializing in mathematical modelling, machine learning, multi-agent architectures, and enterprise AI governance. With 12+ years of experience in AI innovation across AI Consulting, FinTech, Asset Management, and RegTech, he bridges theoretical research with practical enterprise implementation.

- **Website**: [luigisimeone.com](https://luigisimeone.com)
- **LinkedIn**: [linkedin.com/in/luigi-simeone](https://linkedin.com/in/luigi-simeone)

---

*For production deployments, enterprise customization, or consulting engagements, please reach out directly.*
