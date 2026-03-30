# llama-index-readers-pdfmux

[![PyPI version](https://img.shields.io/pypi/v/llama-index-readers-pdfmux.svg)](https://pypi.org/project/llama-index-readers-pdfmux/)
[![Python versions](https://img.shields.io/pypi/pyversions/llama-index-readers-pdfmux.svg)](https://pypi.org/project/llama-index-readers-pdfmux/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

LlamaIndex reader for [pdfmux](https://pdfmux.com) -- self-healing PDF extraction for RAG pipelines.

## Why pdfmux?

Most PDF loaders use a single extraction method and silently fail on complex layouts. pdfmux routes each page through the best extraction pipeline automatically:

- **Smart routing** -- selects the optimal parser per page (text-heavy, scanned, tables, mixed)
- **Confidence scoring** -- every chunk includes a confidence score so your RAG pipeline can filter or re-rank
- **Self-healing** -- retries with alternative extractors when the primary one returns low-quality output

## Install

```bash
pip install llama-index-readers-pdfmux
```

## Usage

```python
from llama_index_readers_pdfmux import PDFMuxReader

reader = PDFMuxReader()
docs = reader.load_data("report.pdf")
```

Each `Document` includes metadata with extraction quality signals:

```python
reader = PDFMuxReader(quality="high")
for doc in reader.load_data("report.pdf"):
    print(doc.metadata)
    # {
    #   "source": "report.pdf",
    #   "title": "Q4 Results",
    #   "page_start": 1,
    #   "page_end": 3,
    #   "tokens": 820,
    #   "confidence": 0.94
    # }
```

### Options

```python
# Quality presets: "fast", "standard" (default), "high"
reader = PDFMuxReader(quality="high")

# Load all PDFs in a directory
docs = reader.load_data("./papers/")

# Custom glob pattern
reader = PDFMuxReader(glob="**/*.pdf")
docs = reader.load_data("./papers/")

# Attach extra metadata
docs = reader.load_data("report.pdf", extra_info={"project": "Q4 analysis"})
```

### With LlamaIndex pipelines

```python
from llama_index.core import VectorStoreIndex
from llama_index_readers_pdfmux import PDFMuxReader

reader = PDFMuxReader(quality="high")
docs = reader.load_data("./papers/")

# Filter low-confidence chunks
docs = [d for d in docs if d.metadata["confidence"] > 0.8]

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("What were the key findings?")
```

## License

MIT
