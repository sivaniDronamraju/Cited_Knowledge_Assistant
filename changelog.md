## Tag - v1.0.0
- Multi-format ingestion (docx, pdf, csv, json, eml)
- Structured row normalization
- Language detection (EN/PL)
- Recursive semantic chunking (NLTK)
- MiniLM multilingual embeddings
- FAISS vector index
- Candidate search + MMR reranking
- Safe index loading
- Integration tests included

## v0.2.0 – Guarded RAG Core

### Added
- FAISS-based vector retrieval
- Manual MMR reranking implementation
- Confidence gating layer
- ContextBuilder for structured LLM context formatting
- PromptBuilder with strict citation enforcement rules
- Unified ResponseValidator (citation + grounding validation)
- Ollama streaming LLM integration (non-API)
- Enterprise QA orchestration layer
- Integration tests for retrieval pipeline
- ContextBuilder unit test

### Refactored
- Removed duplicate validator files
- Merged citation and grounding validation
- Hardened type hints across modules
- Removed redundant embedding normalization
- Improved vector store integrity checks
- Eliminated API coupling from core logic
