# Bilingual HR Bronze (EN/PL) — Synthetic Corporate Data Lake Snapshot (2025-01)

**What this is:** a realistic, *raw/bronze* multi-source corporate snapshot for LLM + RAG + analytics.
All content is synthetic. Not legal advice.

## Contents
- **company_bronze/** (raw, uncleaned)
  - **data/content/policies/** — 17 HR policies × 2 languages (EN/PL) in .docx
  - **data/systems/hris_exports/** — CSV exports (employees, leave balances, training)
  - **data/communications/** — emails (.eml, with attachments) + Slack-style chats (JSON)
  - **data/tickets/** — HR helpdesk (CSV), IT incidents (CSV), compliance cases (JSONL) + evidence
  - **data/files/** — spreadsheets (XLSX), CSV extracts, short PDFs, and a SharePoint-like index (CSV)
- **MANIFEST.csv** — file paths, sizes, sha256 checksums

## Intended use
- Retrieval-Augmented Generation (RAG) over mixed formats
- Cross-lingual search/Q&A (EN ⇄ PL)
- HR analytics exercises (joins across HRIS/training/policy acknowledgements)
- Document AI (chunking, parsing .docx/.eml/.pdf)

## Notes
- Data is intentionally **messy** (duplicates, missing fields, malformed timestamps, malformed emails) to reflect bronze reality.
- Company/people are fictional; numbers are illustrative; no real PII.

## License
**CC BY 4.0** — Please credit the dataset in derived works.

## Version
Release: 2025-01 (v1.0) • Generated with structured prompts (OpenAI) and Python scripts.
