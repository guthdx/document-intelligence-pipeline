# CLAUDE.md - Document Intelligence Pipeline

## Project Overview

Offline-first, document-size-agnostic pipeline to transform scanned medical/research PDFs into a queryable, provenance-preserving corpus.

**Target Platform:** M3 Ultra Mac Studio (512GB unified memory) at 192.168.11.21

## Core Constraints (MUST ENFORCE)

| Constraint | Requirement |
|------------|-------------|
| **Document-Size Agnostic** | No operation requires fitting whole document into memory/context |
| **Bounded-Context LLM** | Max 10-page window for cross-page operations; never whole-document |
| **Offline-First** | All processing local. No cloud API calls. PHI/PII never leaves machine |
| **Provenance Preservation** | Every value traces to source page/block coordinates |

## Architecture Summary

### Deterministic Stages (No LLM)
1. **Page Render** - PDF to PNG at 300 DPI
2. **Layout Detection** - YOLO/DINO bounding boxes + classes
3. **Layout Understanding** - LayoutLMv3 semantic labels (runs AFTER OCR)
4. **Orientation Detection** - Per-block 0/90/180/270 rotation
5. **Printed OCR** - Tesseract/PaddleOCR/Surya
6. **Handwriting OCR** - TrOCR with confidence flags
7. **Table Extraction** - Table Transformer to relational format
8. **Continuation Detection** - Multi-page table/section markers

### LLM Stages (Bounded Context)
1. **Text Normalization** - Conservative OCR error correction
   - NEVER modify: numbers, dates, units
   - ALWAYS retain original_text
   - Store diffs with provenance
2. **Entity Extraction** - Dates, lab values, medications, codes
3. **Table Labeling** - Column/row headers, data types
4. **Multi-Page Stitching** - 2-10 page window, evidence-driven only
5. **Page Synthesis** - Single-page summaries

## Development Commands

```bash
# Setup virtual environment
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run CLI
docint process <pdf>
docint batch <directory>
docint search <query>
docint status

# Database (PostgreSQL on M3 Ultra Docker)
# Connection: postgresql+asyncpg://docint:localdev@localhost:5432/docint

# Tests
pytest
pytest --cov=docint

# Linting
ruff check src/
mypy src/
```

## Directory Structure

```
src/docint/
├── cli.py              # Typer CLI
├── config.py           # Pydantic settings
├── models/             # IR schemas (Pydantic)
├── pipeline/           # Deterministic stages
├── llm/                # LLM stages
├── storage/            # SQLAlchemy + Alembic
├── retrieval/          # FTS + Vector + SQL fusion
├── chunking/           # Embedding preparation
└── audit/              # Review queue
```

## Key Files

- `pyproject.toml` - Project config
- `alembic.ini` - Database migrations
- `docker-compose.yml` - PostgreSQL + pgvector
- `docs/adr/` - Architecture Decision Records

## M3 Ultra Specific

```python
# MPS (Metal Performance Shaders) for PyTorch
import torch
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
```

## Retrieval Strategy

1. **FTS** - PostgreSQL tsvector + pg_trgm
2. **Vector** - pgvector with sentence-transformers embeddings
3. **Structured** - SQL queries on entities/tables
4. **Fusion** - Reciprocal Rank Fusion (RRF)

## ADRs (Architecture Decision Records)

- ADR-001: Page- and block-level processing
- ADR-002: Tables as first-class relational data
- ADR-003: Handwriting as separate data class
- ADR-004: Offline-first execution
- ADR-005: Audit-based human review
