# Offline Document Intelligence Pipeline

**Status:** Architecture Definition Phase  
**Scope:** Internal system capability (not a product)  
**Audience:** Systems, ML, and Data Engineering

---

## 1. Problem Statement

We maintain an archive of **several thousand scanned medical and research PDFs**, ranging from a few pages to **400+ pages**, containing a mix of:

- Printed narrative text
- Tables (labs, vitals, results)
- Structured and semi-structured forms
- Handwritten annotations, including:
  - inline handwritten values
  - margin notes
  - rotated or vertically written text
- Stamps, signatures, and other non-textual marks

These documents are currently **not queryable, not structured, and not reliably searchable**. Manual review is expensive and does not scale.

The goal is to transform this archive into a **provenance-preserving, queryable corpus** while maintaining **offline operation** and **PHI/PII safety**.

---

## 2. Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 16 with pgvector extension
- Tesseract OCR
- Ollama (for local LLM inference)

### Setup

```bash
# Clone repository
git clone git@github.com:guthdx/document-intelligence-pipeline.git
cd document-intelligence-pipeline

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### Database Setup

```bash
# Start PostgreSQL (Docker)
docker compose up -d

# Run migrations
alembic upgrade head
```

---

## 3. Usage

```bash
# Process a single PDF
docint process /path/to/document.pdf

# Batch process directory
docint batch /path/to/pdfs/ --workers 8

# Search corpus
docint search "hemoglobin results"

# Check status
docint status

# Launch audit queue
docint audit
```

---

## 4. Architecture

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

### Core Constraints

| Constraint | Requirement |
|------------|-------------|
| **Document-Size Agnostic** | No operation requires whole document in memory |
| **Bounded-Context LLM** | Max 10-page window for cross-page operations |
| **Offline-First** | All processing local, no cloud APIs |
| **Provenance Preservation** | Every value traces to source coordinates |

### ADRs (Architecture Decision Records)

- [ADR-001: Page- and Block-Level Processing](docs/adr/ADR-001-page-and-block-processing.md)
- [ADR-002: Tables as First-Class Relational Data](docs/adr/ADR-002-tables-first-class.md)
- [ADR-003: Handwriting as Separate Data Class](docs/adr/ADR-003-handwriting-separate-class.md)
- [ADR-004: Offline-First Execution](docs/adr/ADR-004-offline-first.md)
- [ADR-005: Audit-Based Human Review](docs/adr/ADR-005-audit-based-review.md)

---

## 5. Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=docint

# Lint
ruff check src/

# Type check
mypy src/
```

---

## 6. License

MIT License - See [LICENSE](LICENSE) for details.
