# Wikipedia Chunking & Metadata Research

This repository contains an end-to-end experimental pipeline for investigating how Wikipedia text chunking and metadata enrichment affect embedding quality, retrieval performance, and synthetic question generation for RAG-style systems.

The project builds a full pipeline from raw Wikipedia XML → structured chunks → metadata augmentation → embeddings → FAISS indexing → synthetic question generation → retrieval evaluation.

---

# Pipeline Overview

1_raw.py
→ 2_sections.py
→ 3_chunks.py
→ 4_metadata.py
→ 5_embeddings.py / 5_embeddings_vllm.py
→ 6_faiss.py
→ 7_samples.py
→ 8_questions.py / 8_questions_vllm.py
→ 9_refine_vllm.py
→ 10_reword_vllm.py
→ 11_embed_questions.py
→ 12_test.py

Each stage produces artifacts used downstream for embedding and retrieval experiments.

---

# Data Pipeline Stages

## 1. Raw Wikipedia Extraction (`1_raw.py`)

Parses a compressed Wikipedia XML dump (`bz2`) and extracts article content from the main namespace.

**Key features:**
- Streaming XML parsing via `lxml.iterparse`
- Filters:
  - non-article namespaces
  - redirects
  - disambiguation pages

Output: raw_batches/batch_<n>.jsonl

Each record contains:
- `page_title`
- `page_id`
- raw article text

---

## 2. Section Segmentation (`2_sections.py`)

Converts raw article text into structured sections using wiki markup parsing.

**Key features:**
- Uses `mwparserfromhell`
- Removes:
- References
- External links
- boilerplate sections
- Preserves:
- section hierarchy
- section paths
- section depth

Output: section_batches/*.jsonl


Each record includes:
- page metadata
- section title
- hierarchy path
- cleaned section text
- wikilinks

---

## 3. Chunking (`3_chunks.py`)

Splits sections into embedding-friendly chunks.

**Key features:**
- Chunk size: ~800–1600 characters
- Sentence-aware splitting using NLTK
- Preserves section metadata

Output: chunk_parquets/*.parquet


Each chunk includes:
- page + section metadata
- hierarchical path
- chunk text
- lead sentence (when applicable)

---

## 4. Metadata Augmentation (`4_metadata.py`)

Experiments with injecting structured metadata into chunk text.

**Key idea:**
Prefix chunks with structured context such as:
- page title
- section title
- hierarchy depth
- section position
- section path (optional)

Two configurations:
- `NEW_FIELDS`
- `FULL_FIELDS`

Output: chunk_augmented_<test_id>/*.parquet


Each chunk becomes: \[metadata prefix\] + original text


